# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree version written by Xiang Wang
# --------------------------------------------------------

import os
import torch
import numpy as np
import unittest

import sys

from unused import hextree2col
sys.path.append('..')
from modules import HextreeConv, HextreeDeconv, col2hextree
from ..test.utils import get_batch_hextree

# TODO

class TesHextreeConv(unittest.TestCase):

  def test_conv_forward(self):
    r''' Tests hextree2col/col2hextree, hextree_conv/hextree_deconv.
    '''

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/hextree_nn.npz'))
    hextree = get_batch_hextree()

    depth = data['depth'].item()
    in_channels = data['channel'].item()
    out_channels = data['channel_out'].item()
    stride = data['stride']
    kernel_size = data['kernel_size']
    data_in = [torch.from_numpy(data['data_0']),
               torch.from_numpy(data['data_1'])]

    counter = 0
    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        for ne in [True, False]:
          # test hextree2col
          kernel = '{}{}{}'.format(
              kernel_size[j][0], kernel_size[j][1], kernel_size[j][2])
          o2c = hextree2col(
              data_in[ne], hextree, depth, kernel, stride[i], ne)
          gt = data['o2c_%d' % counter]
          self.assertTrue(
              np.array_equal(o2c.numpy(), gt), 'counter: %d' % counter)

          # test col2hextree
          c2o = col2hextree(o2c, hextree, depth, kernel, stride[i], ne)
          gt = data['c2o_%d' % counter]
          self.assertTrue(
              np.array_equal(c2o.numpy(), gt), 'counter: %d' % counter)

          # update counter
          counter = counter + 1

    for m in [True, False]:
      counter = 0
      for i in range(len(stride)):
        for j in range(len(kernel_size)):
          for ne in [True, False]:
            # test hextree_conv
            conv = HextreeConv(
                in_channels, out_channels, kernel_size[j].tolist(), stride[i],
                nempty=ne, direct_method=m, max_buffer=int(2e4))
            weight = torch.from_numpy(data['cw_%d' % counter])
            conv.weights.data.copy_(weight)
            out = conv.forward(data_in[ne], hextree, depth)
            gt = data['conv_%d' % counter]
            self.assertTrue(np.allclose(out.detach().numpy(), gt, atol=1e-6))

            # test hextree_deconv
            deconv = HextreeDeconv(
                in_channels, out_channels, kernel_size[j].tolist(), stride[i],
                nempty=ne, direct_method=m, max_buffer=int(2e4))
            weight = torch.from_numpy(data['dw_%d' % counter])
            deconv.weights.data.copy_(weight)
            out = deconv.forward(data_in[ne], hextree, depth)
            gt = data['dconv_%d' % counter]
            self.assertTrue(np.allclose(out.detach().numpy(), gt, atol=1e-6))

            # update counter
            counter = counter + 1

  def test_conv_backward(self):

    folder = os.path.dirname(__file__)
    data = np.load(os.path.join(folder, 'data/hextree_nn.npz'))
    hextree = get_batch_hextree()

    depth = data['depth'].item()
    in_channels = data['channel'].item()
    out_channels = data['channel_out'].item()
    stride = data['stride']
    kernel_size = data['kernel_size']
    data_in = [torch.from_numpy(data['data_0']),
               torch.from_numpy(data['data_1'])]

    counter = 0
    for i in range(len(stride)):
      for j in range(len(kernel_size)):
        for ne in [True, False]:

          # test hextree_conv
          conv_ref = HextreeConv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, direct_method=True)
          weight = torch.from_numpy(data['cw_%d' % counter])
          conv_ref.weights.data.copy_(weight)
          data_ref = data_in[ne].clone().requires_grad_()
          out_ref = conv_ref.forward(data_ref, hextree, depth)
          loss_ref = out_ref.sum()
          loss_ref.backward()

          conv = HextreeConv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, direct_method=False, max_buffer=int(2e4))
          weight = torch.from_numpy(data['cw_%d' % counter])
          conv.weights.data.copy_(weight)
          data_ = data_in[ne].clone().requires_grad_()
          out = conv.forward(data_, hextree, depth)
          loss = out.sum()
          loss.backward()

          self.assertTrue(np.allclose(
              out.data.numpy(), out_ref.data.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              data_.grad.numpy(), data_ref.grad.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              conv.weights.grad.numpy(), conv_ref.weights.grad.numpy(),
              atol=1e-6))

          # test hextree_deconv
          deconv_ref = HextreeDeconv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, direct_method=True)
          weight = torch.from_numpy(data['dw_%d' % counter])
          deconv_ref.weights.data.copy_(weight)
          data_ref = data_in[ne].clone().requires_grad_()
          out_ref = deconv_ref.forward(data_ref, hextree, depth)
          loss_ref = out_ref.sum()
          loss_ref.backward()

          deconv = HextreeDeconv(
              in_channels, out_channels, kernel_size[j].tolist(), stride[i],
              nempty=ne, direct_method=False, max_buffer=int(2e4))
          weight = torch.from_numpy(data['dw_%d' % counter])
          deconv.weights.data.copy_(weight)
          data_ = data_in[ne].clone().requires_grad_()
          out = deconv.forward(data_, hextree, depth)
          loss = out.sum()
          loss.backward()

          self.assertTrue(np.allclose(
              out.data.numpy(), out_ref.data.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              data_.grad.numpy(), data_ref.grad.numpy(), atol=1e-6))
          self.assertTrue(np.allclose(
              deconv.weights.grad.numpy(), deconv_ref.weights.grad.numpy(),
              atol=1e-6))

          # update counter
          counter = counter + 1


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  unittest.main()
