# M2UMol
Pre-training data, source code, and API for the paper "Multi-to-uni Modal Knowledge Transfer Pre-training for Molecular Representation Learning"

![AppVeyor](https://img.shields.io/badge/python-3.7.10-blue)
![AppVeyor](https://img.shields.io/badge/numpy-1.18.5-red)
![AppVeyor](https://img.shields.io/badge/pytorch-1.7.1-brightgreen)
![AppVeyor](https://img.shields.io/badge/torch--geometric-2.0.0-orange)

<p align="center">
  <img src="pics/overview.pdf" width="80%"/> 
</p>


## Table of Contents
 - [Dataset](#Dataset)
 - [Environment](#Environment)
 - [Pretraining (MolT5-based models)](#pretraining-molt5-based-models)
 - [Finetuning (MolT5-based models)](#finetuning-molt5-based-models)
 - [Datasets](#datasets)
 - [Citation](#citation)

### Environment


### Dataset
 - [Multimodal pre-training dataset]() (txt format)


## Run code
For how to use MRCGNN, we present an example based on the Deng's dataset.

1.Learning drug structural features from drug molecular graphs, you need to change the path in 'drugfeature_fromMG.py' first. If you want use MRCGNN on your own dataset, please ensure the datas in 'trimnet' folds and the datas in 'codes for MRCGNN' folds are the same.)

```
python drugfeature_fromMG.py
```

2.Training/validating/testing for 5 times and get the average scores of multiple metrics.
```
python 5timesrun.py
```

3.You can see the final results of 5 runs in 'test.txt'
