# M2UMol
Pre-training data, source code, and API for the paper "Multi-to-uni Modal Knowledge Transfer Pre-training for Molecular Representation Learning"

![AppVeyor](https://img.shields.io/badge/dive_into_graphs-1.1.0-blue)
![AppVeyor](https://img.shields.io/badge/pytorch-2.0.1-red)
![AppVeyor](https://img.shields.io/badge/transformers-4.39.2-brightgreen)
![AppVeyor](https://img.shields.io/badge/torch_geometric-2.3.1-orange)

<p align="center">
  <img src="pics/overview.png" width="80%"/> 
</p>


## Table of Contents
 - [Environment](#Environment)
 - [Pretraining M2UMol](#Pretraining-M2UMol)
 - [Finetuning M2UMol](#Finetuning-on-three-tasks)
 - [Datasets](#Datasets)
 - [Example of M2UMol as a molecular encoder](#M2UMolencoder)
 - [Molecular analysis API of M2UMol](#Molecular-analysis-API)
 - [Citation](#citation)

### Environment
First, install conda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then create a virtual environment and install packages by using our provided `environment.yml`, and install torch-geometric
```
conda env create -f environment.yml
```
and install torch-geometric and additional dependencies
```
conda activate M2UMol
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
### Pretraining M2UMol
You can first check the 'settings' in 'run_pretrain.py', and modify them according to your needs. You can also set parameters directly in the training command, for example:
```
python run_pretrain.py --lr 0.001 --batch 32 --epochs 150 --mcls_loss_ratio 0.5 --output_name M2UMol
```
For the Text encoder, we utlized pre-trained PubMedBERT by Microsoft. We download pre-trained PubMedBERT at [this Hugging Face link](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/tree/main), and save them in a folder named 'pretrained-PubMedBERT'. For the 3D encoder, we utlized ComENet, a recently proposed method as the encoder, and implement it by using the [dive-into-graphs](https://github.com/divelab/DIG)

Note that the Text encoder and the 3D encoder are all further pre-trained in our pre-training phase, that is, the parameters of these models are not frozen.
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
