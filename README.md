# HexFormer
Task: point cloud sequence segmentation (for now)

## Quick Stark
1. Clone the repository.
```
git clone git@github.com:HerocatUED/HexFormer.git
```
2. Install [Pytorch](https://pytorch.org/) and other requirements (enter the folder *HexFormer*). 
```
cd HexFormer
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```
3. Prepare datasets.
- [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download)
- [HOI4D]()
4. Modify config file and train with 4 nvidia GPUs.
```
python run_seg.py --gpu 0,1,2,3 --alias kitti --port 10008
```

**Note** 
- torch version: function next() used in thsolver.solver;
- Only tested with Python 3.8, torch 2.2.1 with CUDA 11.8


TODO： 
- update to pytorch 2.0, speedup with torch.compile()
- loss design(lower weight for classes that have higher acc)
- CPE: 3D DwConv + 1D Conv
