# TODE-Trans: Transparent Object Depth Estimation with Transformer


[[Paper]](https://arxiv.org/pdf/2209.08455.pdf)


PyTorch implementation of paper "TODE-Trans: Transparent Object Depth Estimation with Transformer"


## Dataset Preparation
### ClearGrasp Dataset
ClearGrasp can be downloaded at their [official website](https://sites.google.com/view/cleargrasp/data) (Both training and testing dataset are needed). After you download zip files and unzip them on your local machine, the folder structure should be like
```
${DATASET_ROOT_DIR}
├── cleargrasp
│   ├── cleargrasp-dataset-train
│   ├── cleargrasp-dataset-test-val
```
### Omniverse Object Dataset
Omniverse Object Dataset can be downloaded [here](https://drive.google.com/drive/folders/1wCB1vZ1F3up5FY5qPjhcfSfgXpAtn31H?usp=sharing). After you download zip files and unzip them on your local machine, the folder structure should be like
```
${DATASET_ROOT_DIR}
├── omniverse
│   ├── train
│   │	├── 20200904
│   │	├── 20200910
```

## TransCG Dataset

TransCG dataset is now available on [official page](https://graspnet.net/transcg). 



## Requirements

The code has been tested under

- Ubuntu 18.04 + NVIDIA GeForce RTX 3090
- PyTorch 1.11.0

System dependencies can be installed by:

```bash
sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
sudo apt install libopenexr-dev zlib1g-dev openexr
```

Other dependencies can be installed by

```bash
pip install -r requirements.txt
```


## Testing
We provide transcg pretrained checkpoints at checkpoints/.

## Training

```
#Train on transcg dataset and test on transcg
python train.py -c ./configs/train_transcg_val_transcg.yaml

#Tran on CGsyn+ood and test on CGreal
python train.py -c ./configs/train_cgsyn+ood_val_cgreal.yaml
#Tran on CGsyn+ood and test on Transcg
python train.py -c ./configs/train_cgsyn+ood_val_transcg.yaml

```


## Citation

```bibtex
@article{2022tode,
    title   = {TODE-Trans: Transparent Object Depth Estimation with Transformer},
    author  = {Kang Chen, Shaochen Wang, Beihao Xia, Dongxu Li, Zhen Kan, and Bin Li},
    journal = {arXiv preprint arXiv:2209.08455}
    year    = {2022}
}
```
