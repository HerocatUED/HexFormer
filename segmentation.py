# build slover for HexFormerSeg

import os
import torch
import numpy as np
from tqdm import tqdm
from thsolver import Solver

import builder
from hextree import Hextree, merge_hextrees, merge_points
from modules import InputFeature


# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


def save_pcd(batch, logit, path, rand_id: float):
    pred = logit.argmax(dim=1)
    # print(f"saving to {path}")
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez(path+'/points_{:.4f}.npz'.format(rand_id),
             batch['points'].points.cpu().numpy())
    np.savez(path+'/label_{:.4f}.npz'.format(rand_id),
             batch['points'].labels.cpu().numpy())
    np.savez(path+'/pred_{:.4f}.npz'.format(rand_id), pred.cpu().numpy())


class SegSolver(Solver):

    def get_model(self, flags):
        return builder.get_segmentation_model(flags)

    def get_dataset(self, flags):
        return builder.get_segmentation_dataset(flags)

    def get_input_feature(self, hextree):
        flags = self.FLAGS.MODEL
        hextree_feature = InputFeature(flags.feature, flags.nempty)
        data = hextree_feature(hextree)
        return data

    def process_batch(self, batch, flags):
        def points2hextree(points):
            hextree = Hextree(flags.depth, flags.full_depth)
            hextree.build_hextree(points)
            return hextree

        if 'hextree' in batch:
            batch['hextree'] = batch['hextree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            query_mask = [pts.points[:, 0] == torch.max(pts.points[:, 0]) for pts in points]
            hextrees = [points2hextree(pts) for pts in points]
            hextree = merge_hextrees(hextrees)
            # hextree.construct_all_neigh()
            batch['points'] = merge_points(points)
            batch['hextree'] = hextree
            batch['query_mask'] = torch.hstack(query_mask)
        return batch

    def model_forward(self, batch):
        hextree, points = batch['hextree'], batch['points']
        data = self.get_input_feature(hextree)
        
        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        query_pts = query_pts[batch['query_mask']]
        logit = self.model(data, hextree, hextree.depth, query_pts)
        label_mask = points.labels > self.FLAGS.LOSS.mask  # filter labels
        return logit[label_mask], points.labels[label_mask]

    def config_optimizer(self):
        flags = self.FLAGS.SOLVER
        if flags.type.lower() == 'adamw_attn':
            base_lr = flags.lr * self.world_size
            transformer_lr_scale = 0.1
            parameters = [
                {"params": [p for n, p in self.model.named_parameters()
                            if "blocks" not in n and p.requires_grad], },
                {"params": [p for n, p in self.model.named_parameters()
                            if "blocks" in n and p.requires_grad],
                 "lr": base_lr * transformer_lr_scale, }, ]
            self.optimizer = torch.optim.AdamW(
                parameters, lr=base_lr, weight_decay=flags.weight_decay)
        else:
            super().config_optimizer()

    def train_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        logit, label = self.model_forward(batch)
        loss = self.loss_function(logit, label)
        accu = self.accuracy(logit, label)
        return {'train/loss': loss, 'train/accu': accu}

    def test_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, label = self.model_forward(batch)
        loss = self.loss_function(logit, label)
        accu = self.accuracy(logit, label)
        num_class = self.FLAGS.LOSS.num_class
        mIoU, IoU = self.IoU_per_class(logit, label, num_class)

        # randomly save 1/100 data for visualization
        rand_id = np.random.uniform()
        if batch['epoch'] % 10 == 0 and batch['epoch'] != 0 and rand_id < 0.01:
            save_pcd(batch, logit, self.logdir+'/result_sample', batch['epoch'])

        names = ['test/loss', 'test/accu', 'test/mIoU'] + \
                ['test/IoU_%d' % i for i in range(num_class)]
        tensors = [loss, accu, mIoU] + IoU
        return dict(zip(names, tensors))

    def eval_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, _ = self.model_forward(batch)
        prob = torch.nn.functional.softmax(logit, dim=1)

        # split predictions
        npts = batch['points'].batch_npt.tolist()
        probs = torch.split(prob, npts)

        # merge predictions
        for i in range(len(probs)):
            prob = probs[i].cpu()
            # Aggregate predictions across different epochs
            filename = 'data' + str(batch['iter_num']) + str(i)
            # self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob

            # Save the prediction results in the last epoch
            if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                full_filename = os.path.join(self.logdir, filename[:-4] + '-eval.npz')
                curr_folder = os.path.dirname(full_filename)
                if not os.path.exists(curr_folder):
                    os.makedirs(curr_folder)
                np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

    def result_callback(self, avg_tracker, epoch):
        r''' Calculate the part mIoU for PartNet and ScanNet.
        '''

        iou_part = 0.0
        avg = avg_tracker.average()

        # Labels smaller than `mask` is ignored. The points with the label 0 in
        # PartNet are background points, i.e., unlabeled points
        mask = self.FLAGS.LOSS.mask + 1
        num_class = self.FLAGS.LOSS.num_class
        for i in range(mask, num_class):
            iou_part += avg['test/IoU_%d' % i]
        iou_part = iou_part / (num_class - mask)

        avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})
        tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))

    def loss_function(self, logit, label):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, label.long())
        return loss

    def accuracy(self, logit, label):
        pred = logit.argmax(dim=1)
        accu = pred.eq(label).float().mean()
        return accu

    def IoU_per_class(self, logit, label, class_num):
        pred = logit.argmax(dim=1)

        mIoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
        IoU, intsc, union = [None] * class_num, [None] * class_num, [None] * class_num
        for k in range(class_num):
            pk, lk = pred.eq(k), label.eq(k)
            intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
            union[k] = torch.sum(torch.logical_or(pk, lk).float())
            IoU[k] = intsc[k] / (union[k] + esp)

            valid = torch.sum(lk.any()) > 0
            valid_part_num += valid.item()
            mIoU += valid * intsc[k] / (union[k] + esp)

        # Calculate the shape IoU for ShapeNet
        mIoU /= valid_part_num + esp
        return mIoU, IoU


if __name__ == "__main__":
    SegSolver.main()
