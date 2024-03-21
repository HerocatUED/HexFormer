# build models

from data_utils import get_hoi4d_seg_dataset, get_kitti_seg_dataset
from hexformerseg import HexFormerSeg


def hexsegformer_large(in_channels, out_channels, init_depth, **kwargs):
    return HexFormerSeg(
        in_channels, out_channels,
        channels=[192, 384, 768, 768],
        num_blocks=[2, 2, 18, 2],
        num_heads=[12, 24, 48, 48],
        patch_size=32, dilation=4,
        drop_path=0.5, nempty=True,
        stem_down=2, head_up=2,
        fpn_channel=168,
        head_drop=[0.5, 0.5], 
        init_depth=init_depth)


def hexsegformer(in_channels, out_channels, init_depth, **kwargs):
    return HexFormerSeg(
        in_channels, out_channels,
        channels=[96, 192, 384, 384],
        num_blocks=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 24],
        patch_size=32, dilation=4,
        drop_path=0.5, nempty=True,
        stem_down=2, head_up=2,
        fpn_channel=168,
        head_drop=[0.5, 0.5], 
        init_depth=init_depth)


def hexsegformer_small(in_channels, out_channels, init_depth, **kwargs):
    return HexFormerSeg(
        in_channels, out_channels,
        channels=[96, 192, 384, 384],
        num_blocks=[2, 2, 6, 2],
        num_heads=[6, 12, 24, 24],
        patch_size=32, dilation=4,
        drop_path=0.5, nempty=True,
        stem_down=2, head_up=2,
        fpn_channel=168,
        head_drop=[0.5, 0.5], 
        init_depth=init_depth)
    
def hexsegformer_toy(in_channels, out_channels, init_depth, **kwargs):
    return HexFormerSeg(
        in_channels, out_channels,
        channels=[32, 48, 64, 128],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=64, dilation=4,
        drop_path=0.3, nempty=True,
        stem_down=2, head_up=2,
        fpn_channel=96,
        head_drop=[0.5, 0.5], 
        init_depth=init_depth)
    

def get_segmentation_model(flags):
    params = {
        'in_channels': flags.channel, 'out_channels': flags.nout,
        'interp': flags.interp, 'nempty': flags.nempty,
        'init_depth': flags.depth
    }
    networks = {
        'hexsegformer': hexsegformer,
        'hexsegformer_large': hexsegformer_large,
        'hexsegformer_small': hexsegformer_small,
        'hexsegformer_toy': hexsegformer_toy,
    }

    return networks[flags.name.lower()](**params)


def get_segmentation_dataset(flags):
    if flags.name.lower() == 'hoi4d':
        return get_hoi4d_seg_dataset(flags)
    elif flags.name.lower() == 'kitti':
        return get_kitti_seg_dataset(flags)
    else:
        raise ValueError
