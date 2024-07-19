from .hoi4d_SemSeg import get_hoi4d_seg_dataset
from .kitti_SemSeg import get_kitti_seg_dataset
from .hoi4d_ActSeg import get_hoi4d_action_seg_dataset

__all__ = [
    "get_hoi4d_seg_dataset",
    "get_kitti_seg_dataset",
    "get_hoi4d_action_seg_dataset",
]

classes = __all__
