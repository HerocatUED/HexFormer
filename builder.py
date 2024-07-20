# build models

from data_utils import (get_hoi4d_sem_seg_dataset, 
                        get_kitti_sem_seg_dataset, 
                        get_hoi4d_act_seg_dataset)
from hexformerSemSeg import HexFormerSemSeg
from hexformerActSeg import HexFormerActSeg


def hexsegformer_large(in_channels, out_channels, **kwargs):
    return HexFormerSemSeg(
        in_channels,
        out_channels,
        channels=[128, 256, 512, 512],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=192,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=256,
        head_drop=[0.5, 0.5],
    )
    

def hexsegformer_hoi4d(in_channels, out_channels, **kwargs):
    return HexFormerSemSeg(
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

def hexsegformer_hoi4d_large(in_channels, out_channels, **kwargs):
    return HexFormerSemSeg(
        in_channels,
        out_channels,
        channels=[128, 256, 512, 1024],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=256,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=256,
        head_drop=[0.5, 0.5],
    )

def hexsegformer(in_channels, out_channels, **kwargs):
    return HexFormerSemSeg(
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
        fpn_channel=128,
        head_drop=[0.5, 0.5],
    )
    
    
def hexsegformer_small(in_channels, out_channels, **kwargs):
    return HexFormerSemSeg(
        in_channels,
        out_channels,
        channels=[32, 64, 128, 128],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=64,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        fpn_channel=64,
        head_drop=[0.5, 0.5],
    )
    
def hexsegformer_action_toy(in_channels, out_channels, **kwargs):
    return HexFormerActSeg(
        in_channels,
        out_channels,
        channels=[32, 64, 128, 128],
        num_blocks=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        patch_size=64,
        dilation=4,
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        head_down=3,
        hid_channel=512,
        head_drop=0.5,
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
        "hexsegformer_hoi4d_large": hexsegformer_hoi4d_large,
        "hexsegformer_action_toy": hexsegformer_action_toy,
    }

    return networks[flags.name.lower()](**params)


def get_segmentation_dataset(flags):
    if flags.name.lower() == "hoi4d_semseg":
        return get_hoi4d_sem_seg_dataset(flags)
    elif flags.name.lower() == "kitti_semseg":
        return get_kitti_sem_seg_dataset(flags)
    elif flags.name.lower() == "hoi4d_actseg":
        return get_hoi4d_act_seg_dataset(flags)
    else:
        raise NotImplementedError
