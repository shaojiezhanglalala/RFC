# Reviewing the Forgotten Classes for Domain Adaptation of Black-Box Predictors
This repo is the official implementation of ["Reviewing the Forgotten Classes for Domain Adaptation of Black-Box Predictors"].
Our method is termed as **RFC**.

## Datasets
Please download and organize the [datasets](https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md) in this structure:
```
RFC
├── data
    ├── office_home
    │   ├── Art
    │   ├── Clipart
    │   ├── Product
    │   ├── Real World
    ├── office31
    │   ├── amazon
    │   ├── dslr
    │   ├── webcam
    ├── visda17
    │   ├── train
    │   ├── validation 
```

Then generate info files with the following commands:
```
python dev/generate_infos.py --ds office_home
python dev/generate_infos.py --ds office31
python dev/generate_infos.py --ds visda17
```

## Train on Office-Home
```
# train black-box source model on domain A
python train_src_v1.py configs/office_home/src_A/train_src_A.py

# adapt with RFC, from A to C
python train_RFC.py configs/office_home/src_A/RFC_C.py
```

## Train on Office-31
```
# train black-box source model on domain a
python train_src_v1.py configs/office31/src_a/train_src_a.py

# adapt with RFC, from a to d
python train_RFC.py configs/office31/src_a/RFC_d.py
```

## Train on VisDA-2017 
```
# train black-box source model
python train_src_v2.py configs/visda17/train_src.py

# adapt with RFC
python train_visda.py configs/visda17/RFC.py
```

## Acknowledge
Part of the codes are adapted from [BETA](https://github.com/xyupeng/BETA). We thank them for their excellent project.
