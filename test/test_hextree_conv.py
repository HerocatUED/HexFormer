import os
import torch
import unittest
import sys

from utils import get_batch_hextree

sys.path.append("..")
from modules import HextreeConvBnRelu, HextreeDeconvBnRelu


class HextreeConvTest(unittest.TestCase):
    # only dimension check, no value check
    def hextree_conv_xyz(self, device):
        print("Load data and build tree")
        htree = get_batch_hextree(
            data_path="test_data/points.npy",
            label_path=None,
            video_ids=[0, 1],
            n_frame=2,
            depth=8,
            full_depth=4,
        )
        htree = htree.to(device)
        nnum_nempty = htree.nnum_nempty[-1]
        data = torch.rand(nnum_nempty, 5).to(device)
        print("Module testing")
        dims = [5, 16, 32, 64, 128]
        convs = torch.nn.ModuleList(
            [
                HextreeConvBnRelu(dims[i], dims[i + 1], [2], 2, True).to(device)
                for i in range(len(dims) - 1)
            ]
        )
        deconvs = torch.nn.ModuleList(
            [
                HextreeDeconvBnRelu(dims[i], dims[i - 1], [2], 2, True).to(device)
                for i in range(len(dims) - 1, 0, -1)
            ]
        )
        print("Conv")
        depth = htree.depth
        for i in range(len(dims) - 1):
            depth_i = depth - i
            data = convs[i](data, htree, depth_i)
            self.assertTrue(data.size()[0] == htree.nnum_nempty[depth_i - 1])
        print("Deconv")
        depth_min = depth_i - 1
        for i in range(len(dims) - 1):
            depth_i = depth_min + i
            data = deconvs[i](data, htree, depth_i)
            self.assertTrue(data.size()[0] == htree.nnum_nempty[depth_i + 1])

    def test_hextree_conv_xyz(self):
        for device in ["cpu", "cuda"]:
            self.hextree_conv_xyz(device=torch.device(device))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    unittest.main()
