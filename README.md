# HexFormer
Official code for **HexFormer: An Efficient Backbone For Point Cloud Video Understanding**

## Quick Stark
1. Clone the repository.
```
git clone git@github.com:HerocatUED/HexFormer.git
```
2. Install [Pytorch](https://pytorch.org/) and other requirements (enter the folder *HexFormer*). 
    **Note**: we tested codes with Python 3.10, torch 2.3.0 with CUDA 11.8
```
cd HexFormer
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```
3. Prepare datasets, download and unzip.
- [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download)
- [HOI4D](https://www.hoi4d.top/#downLoad)
4. Generate filelist. 

    **alias** is the name of tasks, e.g. kitti_SemSeg, hoi4d_SemSeg, hoi4d_ActSeg. 

    **root_dir** is where you download the dataset, e.g. Path to your dataset will be **root_dir/train1.h5** or **root_dir/dataset/sequences/...**, 

```
python data_utils/dataset.py --alias $alias$ --root_dir $.../SemanticKITTI$
```
5. Modify config file. Config files are placed in the folder **configs**. 
Neccessary modifications:
- **logdir**(line 4) if you start a new experiment.
- **has_label**(line 66) $True$ if you conduct train; $Flase$ if you conduct inference. 
- **location** (line 42, 68)
- **batch_size** (line 70) 1 if you conduct inference, otherwise whatever.
- **num_workers** (line 72) 1 if you conduct inference, otherwise whatever.
6. Train.
```
python run.py --run train --alias $alias$ --gpu 0,1,2,3 --port 10008
```
7. Inference. **Note: modify config file before you conduct inference.**
```
python run.py --run test --alias $alias$ --gpu 0 --port 10008 --ckpt $path_to_your_model$
```
After that, you can prepare results for submition according to the task. We provide scripts to convert. This scripts also provide other function you may need for further study, details see **data_utils/tools.py**. An example usage (you can find **predict.npy** under **log_dir** after you run following command):
```
python data_utils/tools.py --task npy_act --log_dir logs/log_3Dconv_4Dattention_CPE_RPE_d9_3blocks_huge_test_hoi4d_actseg
```


TODO： 
- Efficiency Exp
- speedup with torch.compile()
- loss design
- reuse history prediction
- clean locations and paths for datasets
- update config files
- clean the code
