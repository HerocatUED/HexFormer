# HexFormer
Intro: a general 4D backbone

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
3. Prepare datasets, download and unzip.
- [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download)
- [HOI4D](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcFFGX2VfYnctVVNnaU9CSW5Ga0dxR1p4ZU1lP2U9eGFQcGl3&id=12E5C3DBEFFD0594%21291&cid=12E5C3DBEFFD0594)
4. Generate filelist and Modify config file(Take kitti for example)
```
python data_utils/tools.py --dataset kitti --root_dir $.../SemanticKITTI$
```
5. Train
```
python run_seg.py --run train --gpu 0,1,2,3 --alias kitti --port 10008
```
6. Inference(must with only one GPU)
modify config.json, test: batch size = 1, num_worker = 1
```
python run_seg.py --run test --gpu 0 --alias kitti --port 10008 --ckpt $path_to_your_model$
```
Inference speed on KITTI with single 4090GPU: 
20351frame / 3706s = 5.5 FPS
FPS_min = 5, FPS_avg = 6, FPS_top = 8. 
**Note** 
- Tested with Python 3.10, torch 2.3.0 with CUDA 11.8

TODO： 
- speedup with torch.compile()
- loss design
- reuse history prediction
- clean locations and paths
- update config files
- KITTI init feature: polar
- Efficiency Exp
- Add Action Segmentaion task
- HOI4D Vis, center?
