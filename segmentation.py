# build slover for HexFormerSeg

import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from thsolver import Solver

import builder
from hextree import Hextree, merge_hextrees, merge_points
from modules import InputFeature


# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


class SegSolver(Solver):

    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        self.weights = None
        if "kitti" in FLAGS.SOLVER.alias:
            from data_utils.kitti import remap
        elif "hoi4d" in FLAGS.SOLVER.alias:
            from data_utils.hoi4d import remap
        self.remap = remap

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

        if "hextree" in batch:
            batch["hextree"] = batch["hextree"].cuda(non_blocking=True)
            batch["points"] = batch["points"].cuda(non_blocking=True)
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch["points"]]
            query_mask = [
                pts.points[:, 0] == torch.max(pts.points[:, 0]) for pts in points
            ]
            hextrees = [points2hextree(pts) for pts in points]
            hextree = merge_hextrees(hextrees)
            # hextree.construct_all_neigh()
            batch["points"] = merge_points(points)
            batch["hextree"] = hextree
            batch["query_mask"] = torch.hstack(query_mask)
        return batch

    def model_forward(self, batch, train: bool = True):
        hextree, points = batch["hextree"], batch["points"]
        data = self.get_input_feature(hextree)

        query_pts = torch.cat([points.points, points.batch_id], dim=1)
        query_pts = query_pts[batch["query_mask"]]
        logit = self.model(data, hextree, hextree.depth, query_pts)
        if train:
            query_label = points.labels[batch["query_mask"]]
            label_mask = query_label > self.FLAGS.LOSS.mask  # filter labels
            return logit[label_mask], query_label[label_mask]
        else: return logit
        
    def config_optimizer(self):
        flags = self.FLAGS.SOLVER
        if flags.type.lower() == "adamw_attn":
            base_lr = flags.lr * self.world_size
            transformer_lr_scale = 0.1
            parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "blocks" not in n and p.requires_grad
                    ],
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "blocks" in n and p.requires_grad
                    ],
                    "lr": base_lr * transformer_lr_scale,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                parameters, lr=base_lr, weight_decay=flags.weight_decay
            )
        else:
            super().config_optimizer()

    def train_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        logit, label = self.model_forward(batch)
        loss = self.loss_function(logit, label)
        accu = self.accuracy(logit, label)
        return {"train/loss": loss, "train/accu": accu}

    def test_step(self, batch): # this is evaluation when training
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, label = self.model_forward(batch)
        loss = self.loss_function(logit, label)
        accu = self.accuracy(logit, label)
        num_class = self.FLAGS.LOSS.num_class
        mIoU, insc, union = self.IoU_per_class(logit, label, num_class)
        names = (
            ["test/loss", "test/accu", "test/mIoU"]
            + ["test/intsc_%d" % i for i in range(num_class)]
            + ["test/union_%d" % i for i in range(num_class)]
        )
        tensors = [loss, accu, mIoU] + insc + union
        return dict(zip(names, tensors))

    def eval_step(self, batch): # this is test when inference
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        origin_filename = batch['filename'][0]
        root_pos = origin_filename.find("/velodyne")
        scan_id = origin_filename[root_pos-2: root_pos]
        frame_id = origin_filename[root_pos+10: root_pos+16]
        filename = self.logdir + f"/sequences/{scan_id}/predictions/{frame_id}.label"
        curr_folder = os.path.dirname(filename)
        if not os.path.exists(curr_folder):
            os.makedirs(curr_folder)
        with torch.no_grad():
            logit = self.model_forward(batch, False)
        pred = logit.argmax(dim=1).cpu().numpy().astype(np.int32)
        pred = self.remap(pred, True)
        pred.tofile(filename)
        
        
    def result_callback(self, avg_tracker, epoch):
        r"""Calculate the part mIoU."""

        iou_part = 0.0
        avg = avg_tracker.average()

        # Labels smaller than `mask` is ignored. The points with the label 0 in
        # KITTI are background points, i.e., unlabeled points
        mask = self.FLAGS.LOSS.mask + 1
        num_class = self.FLAGS.LOSS.num_class
        for i in range(mask, num_class):
            instc_i = avg["test/intsc_%d" % i]
            union_i = avg["test/union_%d" % i]
            iou_part += instc_i / (union_i + 1.0e-10)
        iou_part = iou_part / (num_class - mask)

        avg_tracker.update({"test/mIoU_part": torch.Tensor([iou_part])})
        tqdm.write("=> Epoch: %d, test/mIoU_part: %f" % (epoch, iou_part))

    def loss_function(self, logit, label):
        class_weight = None
        if self.FLAGS.LOSS.weighted:
            class_weight = self.get_weight()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        loss = criterion(logit, label.long())
        return loss

    def accuracy(self, logit, label):
        pred = logit.argmax(dim=1)
        accu = pred.eq(label).float().mean()
        return accu

    def IoU_per_class(self, logit, label, class_num):
        pred = logit.argmax(dim=1)

        mIoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
        intsc, union = [None] * class_num, [None] * class_num
        for k in range(class_num):
            pk, lk = pred.eq(k), label.eq(k)
            intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
            union[k] = torch.sum(torch.logical_or(pk, lk).float())

            valid = torch.sum(lk.any()) > 0
            valid_part_num += valid.item()
            mIoU += valid * intsc[k] / (union[k] + esp)

        # Calculate the mIoU
        mIoU /= valid_part_num + esp
        return mIoU, intsc, union

    def get_weight(self):
        """
        Get weights for weighted CrossEntropyLoss, only used in SemanticKITTI.
        """
        if self.weights is not None:
            return self.weights
        DATA = yaml.safe_load(open("config/kitti/semantic-kitti-all.yaml", "r"))
        remapdict = DATA["learning_map_inv"]
        # make lookup table for mapping
        maxkey = max(remapdict.keys())
        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        labels = remap_lut[np.arange(1, 26)]
        # content
        content = DATA["content"]
        content_lut = np.zeros((300), dtype=np.float32)
        content_lut[list(content.keys())] = list(content.values())
        weight = np.zeros(26)
        weight[1:] = np.array(content_lut[labels])
        weight = weight / weight.sum()
        weight[0] = 1  # log 0 will be a bug
        weight = -np.log(weight)
        # weight = np.clip(1 / (weight + 1e-10), 0, self.FLAGS.LOSS.weight_clip)
        self.weights = torch.tensor(weight, dtype=torch.float32).cuda()
        return self.weights


if __name__ == "__main__":
    SegSolver.main()
