# build models

from data_utils import get_hoi4d_seg_dataset, get_kitti_seg_dataset
from hexformerseg import HexFormerSeg


def hexsegformer_large(in_channels, out_channels, **kwargs):
    return HexFormerSeg(
        in_channels,
        out_channels,
        channels=[128, 256, 512, 512],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=128,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=512,
        head_drop=[0.5, 0.5],
    )
    

def hexsegformer_hoi4d(in_channels, out_channels, **kwargs):
    return HexFormerSeg(
        in_channels,
        out_channels,
        channels=[128, 256, 512, 512],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=128,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=256,
        head_drop=[0.5, 0.5],
    )


def hexsegformer(in_channels, out_channels, **kwargs):
    return HexFormerSeg(
        in_channels,
        out_channels,
        channels=[64, 128, 256, 256],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=128,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=256,
        head_drop=[0.5, 0.5],
    )
    
    
def hexsegformer_small(in_channels, out_channels, **kwargs):
    return HexFormerSeg(
        in_channels,
        out_channels,
        channels=[32, 64, 128, 128],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=128,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=128,
        head_drop=[0.5, 0.5],
    )


def get_segmentation_model(flags):
    params = {
        "in_channels": flags.channel,
        "out_channels": flags.nout,
        "interp": flags.interp,
        "nempty": flags.nempty,
    }
    networks = {
        "hexsegformer": hexsegformer,
        "hexsegformer_large": hexsegformer_large,
        "hexsegformer_small": hexsegformer_small,
        # "hexsegformer_toy": hexsegformer_toy,
    }

    return networks[flags.name.lower()](**params)


def get_segmentation_dataset(flags):
    if flags.name.lower() == "hoi4d":
        return get_hoi4d_seg_dataset(flags)
    elif flags.name.lower() == "kitti":
        return get_kitti_seg_dataset(flags)
    else:
        raise ValueError
