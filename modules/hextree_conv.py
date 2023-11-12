# --------------------------------------------------------
# Hextree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# Hextree version written by Xiang Wang
# --------------------------------------------------------

import torch
import torch.nn
from torch.autograd import Function
from typing import List

from .hextree_pad import hextree_pad, hextree_depad
from .hextree2col import hextree2col, col2hextree
# import sys
# sys.path.append('..')
from hextree.utils import scatter_add, xavier_uniform_, resize_with_last_val, list2str
from hextree import Hextree


class HextreeConvBase:

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False, max_buffer: int = int(2e8)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = resize_with_last_val(kernel_size)
        self.kernel = list2str(self.kernel_size)
        self.stride = stride
        self.nempty = nempty
        self.max_buffer = max_buffer  # about 200M

        self.kdim = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.kernel_size[3]
        self.in_conv = in_channels if self.is_conv_layer() else out_channels
        self.out_conv = out_channels if self.is_conv_layer() else in_channels
        self.weights_shape = (self.kdim, self.in_conv, self.out_conv)

    def is_conv_layer(self):
        r''' Returns :obj:`True` to indicate this is a convolution layer.
        '''

        raise NotImplementedError

    def setup(self, hextree: Hextree, depth: int):
        r''' Setup the shapes of each tensor.
        This function MUST be called before :obj:`forward_gemm`, :obj:`backward_gemm`
        and :obj:`weight_gemm`.
        '''

        # The depth of tensors:
        # The in_depth and out_depth are the hextree depth of the input and output
        # data; neigh_depth is the hextree depth of the neighborhood information, as
        # well as `col` data, neigh_depth is always the same as the depth of larger
        # data when doing hextree2col or col2hextree.
        self.in_depth = depth
        self.out_depth = depth
        self.neigh_depth = depth
        if self.stride == 2:
            if self.is_conv_layer():
                self.out_depth = depth - 1
            else:
                self.out_depth = depth + 1
                self.neigh_depth = depth + 1

        # The height of tensors
        if self.nempty:
            self.in_h = hextree.nnum_nempty[self.in_depth]
            self.out_h = hextree.nnum_nempty[self.out_depth]
        else:
            self.in_h = hextree.nnum[self.in_depth]
            self.out_h = hextree.nnum[self.out_depth]
            if self.stride == 2:
                if self.is_conv_layer():
                    self.out_h = hextree.nnum_nempty[self.out_depth]
                else:
                    self.in_h = hextree.nnum_nempty[self.in_depth]
        self.in_shape = (self.in_h, self.in_channels)
        self.out_shape = (self.out_h, self.out_channels)

        # The neighborhood indices
        self.neigh = hextree.get_neigh(
            self.neigh_depth, self.kernel, self.stride, self.nempty)

        # The heigh and number of the temporary buffer
        self.buffer_n = 1
        self.buffer_h = self.neigh.shape[0]
        ideal_size = self.buffer_h * self.kdim * self.in_conv
        if ideal_size > self.max_buffer:
            kc = self.kdim * self.in_conv            # make `max_buffer` be divided
            max_buffer = self.max_buffer // kc * kc  # by `kc` with no remainder
            self.buffer_n = (ideal_size + max_buffer - 1) // max_buffer
            self.buffer_h = (self.buffer_h + self.buffer_n - 1) // self.buffer_n
        # self.buffer_shape = (self.buffer_h, self.kdim, self.in_conv)
        self.buffer_shape = (self.buffer_h, self.kdim, 2, self.in_conv) # 2 indicates shape of shuffled key

    def check_and_init(self, data: torch.Tensor):
        r''' Checks the input data and initializes the shape of output data.
        '''

        # Check the shape of input data
        check = tuple(data.shape) == self.in_shape
        assert check, 'The shape of input data is wrong.'

        # Init the output data
        out = data.new_zeros(self.out_shape)
        return out

    def forward_gemm(self, out: torch.Tensor, data: torch.Tensor,
                     weights: torch.Tensor):
        r''' Peforms the forward pass of hextree-based convolution.
        '''

        # Initialize the buffer
        buffer = data.new_empty(self.buffer_shape)

        # Loop over each sub-matrix
        for i in range(self.buffer_n):
            start = i * self.buffer_h
            end = (i + 1) * self.buffer_h

            # The boundary case in the last iteration
            if end > self.neigh.shape[0]:
                dis = end - self.neigh.shape[0]
                end = self.neigh.shape[0]
                buffer, _ = buffer.split([self.buffer_h-dis, dis])

            # Perform hextree2col
            neigh_i = self.neigh[start:end]
            valid = neigh_i >= 0
            buffer.fill_(0)
            # print('valid',valid.shape)
            # print('neigh',neigh_i.shape)
            # print('data',data.shape)
            # print('buffer',buffer.shape)
            buffer[valid] = data[neigh_i[valid]]

            # The sub-matrix gemm
            out[start:end] = torch.mm(buffer.flatten(1, 2), weights.flatten(0, 1))

        return out

    def backward_gemm(self, out: torch.Tensor, grad: torch.Tensor,
                      weights: torch.Tensor):
        r''' Performs the backward pass of hextree-based convolution.
        '''

        # Loop over each sub-matrix
        for i in range(self.buffer_n):
            start = i * self.buffer_h
            end = (i + 1) * self.buffer_h

            # The boundary case in the last iteration
            if end > self.neigh.shape[0]:
                end = self.neigh.shape[0]

            # The sub-matrix gemm
            buffer = torch.mm(grad[start:end], weights.flatten(0, 1).t())
            buffer = buffer.view(-1,
                                 self.buffer_shape[1], self.buffer_shape[2])
            buffer = buffer.to(out.dtype)  # for pytorch.amp

            # Performs col2hextree
            neigh_i = self.neigh[start:end]
            valid = neigh_i >= 0
            out = scatter_add(buffer[valid], neigh_i[valid], dim=0, out=out)

        return out

    def weight_gemm(
            self, out: torch.Tensor, data: torch.Tensor, grad: torch.Tensor):
        r''' Computes the gradient of the weight matrix.
        '''

        # Record the shape of out
        out_shape = out.shape
        out = out.flatten(0, 1)

        # Initialize the buffer
        buffer = data.new_empty(self.buffer_shape)

        # Loop over each sub-matrix
        for i in range(self.buffer_n):
            start = i * self.buffer_h
            end = (i + 1) * self.buffer_h

            # The boundary case in the last iteration
            if end > self.neigh.shape[0]:
                d = end - self.neigh.shape[0]
                end = self.neigh.shape[0]
                buffer, _ = buffer.split([self.buffer_h-d, d])

            # Perform hextree2col
            neigh_i = self.neigh[start:end]
            valid = neigh_i >= 0
            buffer.fill_(0)
            buffer[valid] = data[neigh_i[valid]]

            # Accumulate the gradient via gemm
            out.addmm_(buffer.flatten(1, 2).t(), grad[start:end])

        return out.view(out_shape)


class _HextreeConv(HextreeConvBase):
    r''' Instantiates _HextreeConvBase by overriding `is_conv_layer`
    '''

    def is_conv_layer(self): return True


class _HextreeDeconv(HextreeConvBase):
    r''' Instantiates _HextreeConvBase by overriding `is_conv_layer`
    '''

    def is_conv_layer(self): return False


class HextreeConvFunction(Function):
    r''' Wrap the hextree convolution for auto-diff.
    '''

    @staticmethod
    def forward(ctx, data: torch.Tensor, weights: torch.Tensor, hextree: Hextree,
                depth: int, in_channels: int, out_channels: int,
                kernel_size: List[int] = [3, 3, 3, 3], stride: int = 1,
                nempty: bool = False, max_buffer: int = int(2e8)):
        hextree_conv = _HextreeConv(
            in_channels, out_channels, kernel_size, stride, nempty, max_buffer)
        hextree_conv.setup(hextree, depth)
        out = hextree_conv.check_and_init(data)
        out = hextree_conv.forward_gemm(out, data, weights)

        ctx.save_for_backward(data, weights)
        ctx.hextree_conv = hextree_conv
        return out

    @staticmethod
    def backward(ctx, grad):
        data, weights = ctx.saved_tensors
        hextree_conv = ctx.hextree_conv

        grad_out = None
        if ctx.needs_input_grad[0]:
            grad_out = torch.zeros_like(data)
            grad_out = hextree_conv.backward_gemm(grad_out, grad, weights)

        grad_w = None
        if ctx.needs_input_grad[1]:
            grad_w = torch.zeros_like(weights)
            grad_w = hextree_conv.weight_gemm(grad_w, data, grad)

        return (grad_out, grad_w) + (None,) * 8


class HextreeDeconvFunction(Function):
    r''' Wrap the hextree deconvolution for auto-diff.
    '''

    @staticmethod
    def forward(ctx, data: torch.Tensor, weights: torch.Tensor, hextree: Hextree,
                depth: int, in_channels: int, out_channels: int,
                kernel_size: List[int] = [3, 3, 3, 3], stride: int = 1,
                nempty: bool = False, max_buffer: int = int(2e8)):
        hextree_deconv = _HextreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty, max_buffer)
        hextree_deconv.setup(hextree, depth)
        out = hextree_deconv.check_and_init(data)
        out = hextree_deconv.backward_gemm(out, data, weights)

        ctx.save_for_backward(data, weights)
        ctx.hextree_deconv = hextree_deconv
        return out

    @staticmethod
    def backward(ctx, grad):
        data, weights = ctx.saved_tensors
        hextree_deconv = ctx.hextree_deconv

        grad_out = None
        if ctx.needs_input_grad[0]:
            grad_out = torch.zeros_like(data)
            grad_out = hextree_deconv.forward_gemm(grad_out, grad, weights)

        grad_w = None
        if ctx.needs_input_grad[1]:
            grad_w = torch.zeros_like(weights)
            grad_w = hextree_deconv.weight_gemm(grad_w, grad, data)

        return (grad_out, grad_w) + (None,) * 8


# alias
hextree_conv = HextreeConvFunction.apply
hextree_deconv = HextreeDeconvFunction.apply


class HextreeConv(HextreeConvBase, torch.nn.Module):
    r''' Performs hextree convolution.

    Args:
      in_channels (int): Number of input channels.
      out_channels (int): Number of output channels.
      kernel_size (List(int)): The kernel shape, choose from :obj:`[3]`, :obj:`[2]`,
          :obj:`[3,3,3,3]`, :obj:`[3,1,1,1]`, :obj:`[1,3,1,1]`, :obj:`[1,1,3,1]`, :obj:`[1,1,1,3]`,
          :obj:`[2,2,2,2]`, :obj:`[3,3,1,1]`, :obj:`[1,3,3,1]`, and :obj:`[1,1,3,3]`.
      stride (int): The stride of the convolution (:obj:`1` or :obj:`2`).
      nempty (bool): If True, only performs the convolution on non-empty
          hextree nodes.
      direct_method (bool): If True, directly performs the convolution via using
          gemm and hextree2col/col2hextree. The hextree2col/col2hextree needs to
          construct a large matrix, which may consume a lot of memory. If False,
          performs the convolution in a sub-matrix manner, which can save the
          requied runtime memory.
      use_bias (bool): If True, add a bias term to the convolution.
      max_buffer (int): The maximum number of elements in the buffer, used when
          :attr:`direct_method` is False.

    .. note::
      Each non-empty hextree node has exactly 8 children nodes, among which some
      children nodes are non-empty and some are empty. If :attr:`nempty` is true,
      the convolution is performed on non-empty hextree nodes only, which is exactly
      the same as SparseConvNet and MinkowsiNet; if :attr:`nempty` is false, the
      convolution is performed on all hextree nodes, which is essential for shape
      reconstruction tasks and can also be used in classification and segmentation
      (with slightly better performance and larger memory cost).
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False, direct_method: bool = False,
                 use_bias: bool = False, max_buffer: int = int(2e8)):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, nempty, max_buffer)

        self.direct_method = direct_method
        self.use_bias = use_bias
        self.weights = torch.nn.Parameter(torch.Tensor(*self.weights_shape))
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weights)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def is_conv_layer(self): return True

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''' Defines the hextree convolution.

        Args:
          data (torch.Tensor): The input data.
          hextree (Hextree): The corresponding hextree.
          depth (int): The depth of current hextree.
        '''

        if self.direct_method:
            col = hextree2col(
                data, hextree, depth, self.kernel, self.stride, self.nempty)
            out = torch.mm(col.flatten(1), self.weights.flatten(0, 1))
        else:
            out = hextree_conv(
                data, self.weights, hextree, depth, self.in_channels,
                self.out_channels, self.kernel_size, self.stride, self.nempty,
                self.max_buffer)

        if self.use_bias:
            out += self.bias

        if self.stride == 2 and not self.nempty:
            out = hextree_pad(out, hextree, depth-1)
        return out

    def extra_repr(self) -> str:
        r''' Sets the extra representation of the module.
        '''

        return ('in_channels={}, out_channels={}, kernel_size={}, stride={}, '
                'nempty={}, bias={}').format(self.in_channels, self.out_channels,
                 self.kernel_size, self.stride, self.nempty, self.use_bias)  # noqa


class HextreeDeconv(HextreeConv):
    r''' Performs hextree deconvolution.

    Please refer to :class:`HextreeConv` for the meaning of the arguments.
    '''

    def is_conv_layer(self): return False

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''' Defines the hextree deconvolution.

        Please refer to :meth:`HextreeConv.forward` for the meaning of the arguments.
        '''

        depth_col = depth
        if self.stride == 2:
            depth_col = depth + 1
            if not self.nempty:
                data = hextree_depad(data, hextree, depth)

        if self.direct_method:
            col = torch.mm(data, self.weights.flatten(0, 1).t())
            col = col.view(col.shape[0], self.kdim, -1)
            out = col2hextree(
                col, hextree, depth_col, self.kernel, self.stride, self.nempty)
        else:
            out = hextree_deconv(
                data, self.weights, hextree, depth, self.in_channels,
                self.out_channels, self.kernel_size, self.stride, self.nempty,
                self.max_buffer)

        if self.use_bias:
            out += self.bias
        return out


# bn_momentum, bn_eps = 0.01, 0.001    # the default value of Tensorflow 1.x
bn_momentum, bn_eps = 0.1, 1e-05   # the default value of pytorch


class HextreeConvBn(torch.nn.Module):
    r''' A sequence of :class:`HextreeConv` and :obj:`BatchNorm`.

    Please refer to :class:`HextreeConv` for details on the parameters.
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False):
        super().__init__()
        self.conv = HextreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''''''

        out = self.conv(data, hextree, depth)
        out = self.bn(out)
        return out


class HextreeConvBnRelu(torch.nn.Module):
    r''' A sequence of :class:`HextreeConv`, :obj:`BatchNorm`, and :obj:`Relu`.

    Please refer to :class:`HextreeConv` for details on the parameters.
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False):
        super().__init__()
        self.conv = HextreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''''''

        out = self.conv(data, hextree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out


class HextreeDeconvBnRelu(torch.nn.Module):
    r''' A sequence of :class:`HextreeDeconv`, :obj:`BatchNorm`, and :obj:`Relu`.

    Please refer to :class:`HextreeDeconv` for details on the parameters.
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [3], stride: int = 1,
                 nempty: bool = False):
        super().__init__()
        self.deconv = HextreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, hextree: Hextree, depth: int):
        r''''''

        out = self.deconv(data, hextree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out
