import os
import torch
import unittest

import sys

sys.path.append("..")
from hextree import key2txyz, txyz2key


class ShuffledKeyTest(unittest.TestCase):

    def test_shuffled_key_rand_case(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for d in devices:
            t = torch.randint(256, (10000,), device=d)
            x = torch.randint(65536, (10000,), device=d)
            y = torch.randint(65536, (10000,), device=d)
            z = torch.randint(65536, (10000,), device=d)
            b = torch.randint(128, (10000,), device=d)

            key = txyz2key(t, x, y, z, b, 16)
            t1, x1, y1, z1, b1 = key2txyz(key, 16)

            self.assertTrue((t1 == t).all())
            self.assertTrue((x1 == x).all())
            self.assertTrue((y1 == y).all())
            self.assertTrue((z1 == z).all())
            self.assertTrue((b1 == b).all())


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    unittest.main()
