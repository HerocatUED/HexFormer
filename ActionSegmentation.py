# build slover for HexFormerSeg

import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from thsolver import Solver

from hextree import Hextree, merge_hextrees, merge_points
from modules import InputFeature
from builder import get_segmentation_dataset, get_segmentation_model

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


class ActSegSolver(Solver):

    def __init__(self, FLAGS, is_master=True):
        super().__init__(FLAGS, is_master)
        if "hoi4d_ActSeg" in FLAGS.SOLVER.alias:
            from data_utils.hoi4d_ActSeg import remap
        else: raise NotImplementedError
        self.remap = remap
        self.overlap = [0.10, 0.25, 0.50]
        self.overlap_len = len(self.overlap)
        self.video_len = 150

    def get_model(self, flags):
        return get_segmentation_model(flags)

    def get_dataset(self, flags):
        return get_segmentation_dataset(flags)

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
        pred = logit.argmax(dim=1)
        score, tp, fp, fn = self.score(pred, label)
        names = (["test/loss", "test/accu", "test/edit_score"]
                 + ["test/tp_%.2f" % o for o in self.overlap]
                 + ["test/fp_%.2f" % o for o in self.overlap]
                 + ["test/fn_%.2f" % o for o in self.overlap])
        tensors = [loss, accu, score] + tp + fp + fn
        return dict(zip(names, tensors))

    def eval_step(self, batch): # this is test when inference
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        origin_filename = batch['filename'][0]
        pos1 = origin_filename.find("/velodyne")
        pos2 = origin_filename.find("/sequences")
        scan_id = origin_filename[pos2+11: pos1]
        frame_id = origin_filename[pos1+10: pos1+16]
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
        r"""Calculate F1 Score.""" 
        
        avg = avg_tracker.average()

        for i in range(self.overlap_len):
            tp_i = avg["test/tp_%.2f" % self.overlap[i]]
            fp_i = avg["test/fp_%.2f" % self.overlap[i]]
            fn_i = avg["test/fn_%.2f" % self.overlap[i]]
            precision = tp_i / float(tp_i + fp_i)
            recall = tp_i / float(tp_i + fn_i)
            f1 = 2.0 * (precision * recall) / (precision + recall + 1.0e-10)
            f1 *= 100
            avg_tracker.update({"test/f1_%.2f" % self.overlap[i]: torch.Tensor([f1])})
            tqdm.write("=> Epoch: %d, test/f1_%.2f" % (epoch, f1))

    def loss_function(self, logit, label):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, label.long())
        return loss

    def accuracy(self, logit, label):
        pred = logit.argmax(dim=1)
        accu = pred.eq(label).float().mean()
        return accu
    
    def score(self, pred, label):
        edit = 0
        tp, fp, fn = torch.zeros(self.overlap_len), torch.zeros(self.overlap_len), torch.zeros(self.overlap_len)
        pred, label = pred.view(-1, self.video_len), label.view(-1, self.video_len)
        for i in range(pred.size(0)):
            for j in range(self.overlap_len):
                tp_, fp_, fn_ = self.f_score(pred[i], label[i], self.overlap[j])
                tp[j] += tp_
                fp[j] += fp_
                fn[j] += fn_
            edit += self.edit_score(pred[i], label[i])
        return edit / pred.size(0), tp, fp, fn
    
    def edit_score(self, pred, label, norm = True, bg_class = ["background"]):
        P, _, _ = self.get_labels_start_end_time(pred, bg_class)
        Y, _, _ = self.get_labels_start_end_time(label, bg_class)
        return self.levenstein(P, Y, norm)
    
    def get_labels_start_end_time(self, frame_wise_labels, bg_class = ["background"]):
        change_points = torch.nonzero(torch.diff(frame_wise_labels, prepend=frame_wise_labels[:1])).flatten()
        starts = torch.cat((torch.tensor([0]), change_points))
        labels = frame_wise_labels[starts]
        ends = torch.cat((change_points, torch.tensor([len(frame_wise_labels) - 1])))
        return labels, starts, ends
 
    def levenstein(self, pred, label, norm = False):
        r"""calculate levenstein distance"""
        m_row = len(pred)
        n_col = len(label)

        # Initialize the matrix
        D = torch.zeros((m_row + 1, n_col + 1), dtype=torch.float32)
        D[:, 0] = torch.arange(m_row + 1, dtype=torch.float32)
        D[0, :] = torch.arange(n_col + 1, dtype=torch.float32)

        pred_expanded = pred.unsqueeze(1).expand(-1, n_col)
        label_expanded = label.unsqueeze(0).expand(m_row, -1)
        match_matrix = (pred_expanded != label_expanded).float()

        # Fill the matrix using dynamic programming
        for i in range(1, m_row + 1):
            for j in range(1, n_col + 1):
                cost = match_matrix[i - 1, j - 1]
                D[i, j] = torch.min(torch.tensor([D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost]))
        if norm:
            score = (1 - D[m_row, n_col] / max(m_row, n_col)) * 100
        else: score = D[m_row, n_col]
        return score
    
    def f_score(self, pred, label, overlap, bg_class=["background"]):
        p_label, p_start, p_end = self.get_labels_start_end_time(pred, bg_class)
        y_label, y_start, y_end = self.get_labels_start_end_time(label, bg_class)
    
        tp = 0
        fp = 0
        hits = np.zeros(len(y_label))
    
        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()
    
            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)


if __name__ == "__main__":
    ActSegSolver.main()
