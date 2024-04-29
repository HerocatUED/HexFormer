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
3. Prepare datasets, download and unzip.
- [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download)
- [HOI4D]()**TODO**
4. Generate filelist and Modify config file(Take kitti for example)
```
python data_utils/tools.py --dataset kitti --root_dir $.../SemanticKITTI$
python data_utils/tools.py --dataset hoi4d
```
5. Train
```
python run_seg.py --run train --gpu 0,1,2,3 --alias kitti --port 10008
```
6. Inference(must with only one GPU)
```
python run_seg.py --run test --gpu 0 --alias kitti --port 10008 --ckpt $path_to_your_model$
```
**Note** 
- torch version: function next() used in thsolver.solver;
- Only tested with Python 3.8, torch 2.2.1 with CUDA 11.8


TODO： 
- Inference Code
- update to pytorch 2.0, speedup with torch.compile()
- loss design
- CPE: 3D DwConv + 1D Conv
- reuse history prediction
- Use corlor infomation as init feature
- FPS: current 1~2 frame/GPU
- Cross Attention/ mask attention: query Current
- encoder num
- inference: Vote
- locations: 
    - HOI4D: float32 in hoi4d.py
    - KITTI and HOI4D: location in config file
