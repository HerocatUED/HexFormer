from .hoi4d import get_hoi4d_seg_dataset
from .kitti import get_kitti_seg_dataset

__all__ = [
    'get_hoi4d_seg_dataset',
    'get_kitti_seg_dataset',
]

classes = __all__