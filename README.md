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
python run_pretrain.py --num_layers 3 --lr 0.001 --batch 32 --epochs 150 --tem 1.0 --mcls_loss_ratio 0.5 --output_name M2UMol
```

For the Text encoder, we utlized a pre-trained large language model (LLM) PubMedBERT proposed by Microsoft. We download pre-trained PubMedBERT at [this Hugging Face link](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/tree/main), and save them in a folder named 'pretrained-PubMedBERT'. For the 3D encoder, we utlized ComENet, a recently proposed method as the encoder, and implement it by using the [dive-into-graphs](https://github.com/divelab/DIG)

Note that the Text encoder and the 3D encoder are all further pre-trained in our pre-training phase, that is, the parameters of these models are not frozen. In addition, you can easily replace different large language models and 3D conformation encoders by modifying the parameters in 'layer.py'.

### Finetuning on three tasks
We comprehensively verified the model performance of M2UMol through three downstream tasks: molecular property prediction, drug-drug interaction prediction and drug-target interaction prediction.

## Molecular property prediction

## Drug-drug interaction prediction
For the drug-drug interaction prediction, you can use the following python script for training and testing for three times with three different seeds (for the cold start split setting, it means three different split):
```
python run.py
```
You can also set different parameters in 'run.py' or run for one time by the following python script:
```
python main.py --seed 0 --lr 0.0005 --batch 256 --weight_decay 0.0002 --dropout 0.7 --split scaffold
```

## Drug-target interaction prediction


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
