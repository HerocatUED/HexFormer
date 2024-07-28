from .hoi4d_SemSeg import get_hoi4d_sem_seg_dataset
from .kitti_SemSeg import get_kitti_sem_seg_dataset
from .hoi4d_ActSeg import get_hoi4d_act_seg_dataset

__all__ = [
    "get_hoi4d_sem_seg_dataset",
    "get_kitti_sem_seg_dataset",
    "get_hoi4d_act_seg_dataset",
]

classes = __all__
