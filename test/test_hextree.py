import os
import torch 
import unittest
import sys 
sys.path.append('..')
import hextree
import ocnn.octree as octree


class TestHextree(unittest.TestCase):

    def init_points(self):
        points = torch.Tensor([[0, -1, -1, -1], [0, 0, 0, -1], [8, 0.0625, 0.0625, -1]])
        octree_pts = [torch.Tensor([[-1, -1, -1], [0, 0, -1]]), 
                      torch.Tensor([[0.0625, 0.0625, -1]])]
        normals = torch.Tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
        octree_feat = [torch.Tensor([[1, -1], [2, -2]]),
                       torch.Tensor([[3, -3]])]
        labels = torch.Tensor([[0], [2], [2]])
        return hextree.Points(points, normals, features, labels), \
            [octree.Points(octree_pts[i], features=octree_feat[i]) for i in range(len(octree_pts))]
    
    def build_hextree(self, device):
        point_cloud, pcds = self.init_points()
        point_cloud = point_cloud.to(device)
        htree = hextree.Hextree(depth=5, full_depth=1, device=device)
        htree.build_hextree(point_cloud)
        htree = htree.to('cpu')
        otrees = []
        for pcd in pcds:
            pcd = pcd.to(device)
            otree = octree.Octree(depth=5, full_depth=1, device=device)
            otree.build_octree(pcd)
            otree = otree.to('cpu')
            otrees.append(otree)
        # test node number
        nnum = torch.Tensor([1, 16, 32, 48, 48, 48])
        nnum_nempty = torch.Tensor([1, 2, 3, 3, 3, 3])
        self.assertTrue((htree.nnum == nnum).all())
        self.assertTrue((htree.nnum_nempty == nnum_nempty).all())

        # test the key
        keys = [
            torch.Tensor([0]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          96, 97, 98, 99, 100, 101, 102, 103, 
                          104, 105, 106, 107, 108, 109, 110, 111]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 
                          1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551,
                          1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 
                          1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          24576, 24577, 24578, 24579, 24580, 24581, 24582, 24583,
                          24584, 24585, 24586, 24587, 24588, 24589, 24590, 24591,
                          26624, 26625, 26626, 26627, 26628, 26629, 26630, 26631, 
                          26632, 26633, 26634, 26635, 26636, 26637, 26638, 26639]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          393216, 393217, 393218, 393219, 393220, 393221, 393222, 393223, 
                          393224, 393225, 393226, 393227, 393228, 393229, 393230, 393231,
                          425984, 425985, 425986, 425987, 425988, 425989, 425990, 425991, 
                          425992, 425993, 425994, 425995, 425996, 425997, 425998, 425999])
        ]
        for d in range(6):
            self.assertTrue((htree.keys[d] == keys[d]).all())

        # test masked_counts
        masked_counts = [
            torch.Tensor([1, 1]),
            torch.Tensor([2, 1]),
            torch.Tensor([1, 1, 1]),
            torch.Tensor([1, 1, 1]),
            torch.Tensor([1, 1, 1]),
            torch.Tensor([1, 1, 1]),
        ]
        for d in range(6):
            self.assertTrue((htree.masked_counts[d] == masked_counts[d]).all())

        # test scatter_idx
        scatter_idx = [
            torch.Tensor([0, 1]),
            torch.Tensor([0, 0, 1]),
            torch.Tensor([0, 1, 2]),
            torch.Tensor([0, 1, 2]),
            torch.Tensor([0, 1, 2]),
            torch.Tensor([0, 1, 2]),
        ]
        for d in range(6):
            self.assertTrue((htree.scatter_idx[d] == scatter_idx[d]).all())

        # test mapping index
        octree_feat = torch.concatenate([otrees[i].features[5] for i in range(len(otrees))])
        hextree_feat = htree.features[5]
        o2h = octree_feat[htree.octree2hextree[5]]
        h20 = hextree_feat[htree.hextree2octree[5]]
        self.assertTrue((o2h == hextree_feat).all())
        self.assertTrue((h20 == octree_feat).all())
        
        # test the children 
        children = [
            torch.Tensor([0]),
            torch.Tensor([0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        ]
        for d in range(6):
            self.assertTrue((htree.children[d] == children[d]).all())
        
        # test the signal
        normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
        self.assertTrue((htree.normals[5] == normals).all())
        self.assertTrue((htree.features[5] == features).all())
    
    def test_build_hextree(self):
        self.build_hextree('cpu')
        if torch.cuda.is_available():
            self.build_hextree('cuda')
    
    def merge_hextree(self, device):
        point_cloud1 = self.init_points()[0].to(device)
        point_cloud2 = self.init_points()[0].to(device)
        htree1 = hextree.Hextree(depth=5, full_depth=1, device=device)
        htree2 = hextree.Hextree(depth=5, full_depth=1, device=device)
        htree1.build_hextree(point_cloud1)
        htree2.build_hextree(point_cloud2)
        htree = hextree.merge_hextrees([htree1, htree2])
        htree = htree.to('cpu')

        nnum = torch.Tensor([1, 16, 32, 48, 48, 48]) * 2
        nnum_nempty = torch.Tensor([1, 2, 3, 3, 3, 3]) * 2
        self.assertTrue((htree.nnum == nnum).all())
        self.assertTrue((htree.nnum_nempty == nnum_nempty).all())

        keys = [
            torch.Tensor([0]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          96, 97, 98, 99, 100, 101, 102, 103, 
                          104, 105, 106, 107, 108, 109, 110, 111]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 
                          1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551,
                          1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 
                          1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          24576, 24577, 24578, 24579, 24580, 24581, 24582, 24583,
                          24584, 24585, 24586, 24587, 24588, 24589, 24590, 24591,
                          26624, 26625, 26626, 26627, 26628, 26629, 26630, 26631, 
                          26632, 26633, 26634, 26635, 26636, 26637, 26638, 26639]),
            torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 
                          9, 10, 11, 12, 13, 14, 15,
                          393216, 393217, 393218, 393219, 393220, 393221, 393222, 393223, 
                          393224, 393225, 393226, 393227, 393228, 393229, 393230, 393231,
                          425984, 425985, 425986, 425987, 425988, 425989, 425990, 425991, 
                          425992, 425993, 425994, 425995, 425996, 425997, 425998, 425999])
        ]
        keys2 = [key.long() | 1 << 56 for key in keys]
        keys = [torch.cat([key.long(), key2]) for key, key2 in zip(keys, keys2)]

        for d in range(6):
            self.assertTrue((htree.keys[d] == keys[d]).all())
        
        masked_counts = [
            torch.Tensor([1, 1, 1, 1]),
            torch.Tensor([2, 1, 2, 1]),
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([1, 1, 1, 1, 1, 1]),
            torch.Tensor([1, 1, 1, 1, 1, 1]),
        ]
        for d in range(6):
            self.assertTrue((htree.masked_counts[d] == masked_counts[d]).all())

        scatter_idx = [
            torch.Tensor([0, 1, 2, 3]),
            torch.Tensor([0, 0, 1, 2, 2, 3]),
            torch.Tensor([0, 1, 2, 3, 4, 5]),
            torch.Tensor([0, 1, 2, 3, 4, 5]),
            torch.Tensor([0, 1, 2, 3, 4, 5]),
            torch.Tensor([0, 1, 2, 3, 4, 5]),
        ]
        for d in range(6):
            self.assertTrue((htree.scatter_idx[d] == scatter_idx[d]).all())

        children = [
            torch.Tensor([0, 1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          2, -1, -1, -1, -1, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1,
                          3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          4, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
            torch.Tensor([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        ]
        for d in range(6):
            self.assertTrue((htree.children[d] == children[d].long()).all())

        normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]).repeat(2, 1)
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]]).repeat(2, 1)
        self.assertTrue((htree.normals[5] == normals).all())
        self.assertTrue((htree.features[5] == features).all())

    def test_merge_hextree(self):
        self.merge_hextree('cpu')
        if torch.cuda.is_available():
            self.merge_hextree('cuda')

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    unittest.main()