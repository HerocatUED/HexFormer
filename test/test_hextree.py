import os
import torch 
import unittest
import sys 
sys.path.append('..')
import hextree
import ocnn.octree as octree

# NOTE: special test, we test hextree with octree.
# accuracy is garenteed by octree.

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
        point_cloud, _ = self.init_points()
        point_cloud = point_cloud.to(device)
        htree = hextree.Hextree(depth=5, full_depth=1, device=device)
        htree.build_hextree(point_cloud)
        htree = htree.to('cpu')
        
        for d in range(htree.depth, htree.full_depth, -1):
            # test node number
            self.assertTrue(htree.nnum[d] == htree.octrees.nnum[d])
            self.assertTrue(htree.nnum_nempty[d] == htree.octrees.nnum_nempty[d])
            self.assertTrue(htree.nnum[d] == htree.children[d].size()[0])

            # test the key and mapping (accually testing children at the same time)
            # key(depth, nempty) used attr: children to get nempty mask
            hkey_all = htree.keys[d]
            okey_all = htree.octrees.keys[d]
            hokey_all = self.key_trans(hkey_all, htree.hex2oct[d])
            self.assertTrue((okey_all == hokey_all).all())
            hkey = htree.key(d, True)
            okey = htree.octrees.key(d, True)
            hokey = self.key_trans(hkey, htree.hex2oct_nempty[d])
            self.assertTrue((okey == hokey).all())
            
        # test the signal
        normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]])
        self.assertTrue((htree.normals[5] == normals).all())
        self.assertTrue((htree.features[5] == features).all())
    
    def key_trans(self, hkey, mapping):
        b = hkey >> 56
        t = hkey & (2 ** 8 - 1)
        okey = hkey >> 8
        okey = okey & (2 ** 48 - 1)
        cnt = 0
        for i in torch.unique(b):
            mask = b == i
            for j in torch.unique(t):
                mask1 = (t == j) & mask
                okey[mask1] = (okey[mask1]) | (cnt << 48)
                cnt += 1
        return okey[mapping]
    
    def test_build_hextree(self):
        self.build_hextree('cpu')
        if torch.cuda.is_available():
            self.build_hextree('cuda')
    
    def merge_hextree(self, device):
        point_cloud1, pcds1 = self.init_points()
        point_cloud1 = point_cloud1.to(device)
        point_cloud2, pcds2 = self.init_points()
        point_cloud2 = point_cloud2.to(device)
        htree1 = hextree.Hextree(depth=5, full_depth=1, device=device)
        htree2 = hextree.Hextree(depth=5, full_depth=1, device=device)
        htree1.build_hextree(point_cloud1)
        htree2.build_hextree(point_cloud2)
        htree = hextree.merge_hextrees([htree1, htree2])
        htree = htree.to('cpu')

        for d in range(htree.depth, htree.full_depth, -1):
            # test node number
            self.assertTrue(htree.nnum[d] == htree.octrees.nnum[d])
            self.assertTrue(htree.nnum_nempty[d] == htree.octrees.nnum_nempty[d])
            self.assertTrue(htree.nnum[d] == htree.children[d].size()[0])

            # test the key and mapping (accually testing children at the same time)
            # key(depth, nempty) used attr: children to get nempty mask
            hkey_all = htree.keys[d]
            okey_all = htree.octrees.keys[d]
            hokey_all = self.key_trans(hkey_all, htree.hex2oct[d])
            self.assertTrue((okey_all == hokey_all).all())
            hkey = htree.key(d, True)
            okey = htree.octrees.key(d, True)
            hokey = self.key_trans(hkey, htree.hex2oct_nempty[d])
            self.assertTrue((okey == hokey).all())
            
        # test the signal
        normals = torch.Tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]).repeat(2, 1)
        features = torch.Tensor([[1, -1], [2, -2], [3, -3]]).repeat(2, 1)
        self.assertTrue((htree.normals[5] == normals).all())
        self.assertTrue((htree.features[5] == features).all())
        # test octrees
        otrees = []
        for pcds in [pcds1, pcds2]:
            for pcd in pcds:
                pcd = pcd.to(device)
                otree = octree.Octree(depth=5, full_depth=1, device=device)
                otree.build_octree(pcd)
                otree = otree.to('cpu')
                otrees.append(otree)
        otrees = octree.merge_octrees(otrees)
        for i in range(1, 6):
            self.assertTrue((otrees.key(i, True) == htree.octrees.key(i, True)).all())

    def test_merge_hextree(self):
        self.merge_hextree('cpu')
        if torch.cuda.is_available():
            self.merge_hextree('cuda')

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    unittest.main()