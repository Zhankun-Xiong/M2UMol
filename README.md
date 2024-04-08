# Multi-to-uni Modal Knowledge Transfer Pre-training for Molecular Representation Learning

![AppVeyor](https://img.shields.io/badge/pytorch-2.0.1-red)
![AppVeyor](https://img.shields.io/badge/torch_geometric-2.3.1-orange)
![AppVeyor](https://img.shields.io/badge/dive_into_graphs-1.1.0-blue)
![AppVeyor](https://img.shields.io/badge/transformers-4.39.2-brightgreen)

## Introduction
This repository contains the pre-training data, source code, and API for the paper "Multi-to-uni Modal Knowledge Transfer Pre-training for Molecular Representation Learning". **M2UMol** is a **M**ulti-**to**-**U**ni modal knowledge transfer pre-training **Mol**ecular representation learning method. M2UMol can be deployed on incomplete multimodal data for efficient pre-training, and can well adapt to downstream tasks with only molecular 2D modality given, which benefits from that M2UMol can generate reliable multimodal representations solely based on molecular 2D topological graphs. It works well on molecular property prediction and molecular interaction prediction tasks, and can also perform key group identification and 2D-to-multimodal retrieval tasks.


## Overview of M2UMol
<p align="center">
  <img src="pics/overview2.png" width="100%"/> 
</p>


## Table of Contents
 - [Environment](#Environment)
 - [Datasets](#Datasets)
 - [Pretraining M2UMol](#Pretraining-M2UMol)
 - [Finetuning M2UMol](#Finetuning-on-three-tasks)
 - [Molecular analysis API of M2UMol](#Molecular-analysis-API-of-M2UMol)
 - [How to use M2UMol in your own project](#How-to-use-M2UMol-in-your-own-project)
 - [Citation](#citation)

## Environment installation
First, install conda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Then create a virtual environment and install packages by using our provided `environment.yml`, and install torch-geometric package：
```
conda env create -f environment.yml
```
and install torch-geometric and additional dependencies
```
conda activate M2UMol
pip install torch_geometric=2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Dataset

### Pre-training dataset
We constructed the [multimodal pre-training dataset]() based on [DrugBank](https://go.drugbank.com/). We first download the full data in XML format from [this link](https://go.drugbank.com/releases/latest#full), and filter out molecules which can not be converted to the 2D molecular graph by RDKit. Then we can download the structure data of molecules in [this link](https://go.drugbank.com/releases/latest#structures), and extract molecular 2D and 3D structure from these data in SDF format, which is the [2Dstructures.sdf] and the [3Dstructures.sdf], respectively. After that, we extract the textual description of molecules from the full data, which is the [description-sup.csv]. Finally we extract the targets and enzymes the molecules can interact/affect and the drug categories the molecules belong to, and obtain lists of all the occurring targets, enzymes and drug categories. For every molecule, we encode it using one-hot encoding based on its association with target, enzyme and drug categories.
| Modality | 2D structure | 3D structure | Textual description | Target | Enzyme | Drug category |
| --- | --- | --- | --- | --- | --- | --- |
| Number | 11571 | 9468 | 6427 | 7124 | 1696 | 7365 |
<p align="center">
  <img src="pics/vene.png" width="30%"/> 
</p>

### Fine-tuning datasets
#### Molecular property prediction
For the molecular property prediction, the datasets can obtained by the following commands:
```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
```
The statistics about the data are as follows:
| Dataset | BBBP | Tox21 | ToxCast | Sider | Clintox | MUV | HIV | BACE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tasks | 1 | 12 | 617 | 27 | 2 | 17 | 1 | 1 |
| Molecules | 2,039 | 7,831 | 8,575 | 1,427 | 1,478 | 93,087 | 41,127 | 1,513 |

#### Drug-drug interaction prediction
For the drug-drug interaction prediction, we utilize the dataset from our previous work [MRCGNN](https://github.com/Zhankun-Xiong/MRCGNN), and applied cold start and scaffold split strategy to split the datasets, the details can be found in 'split.py'. After that, we obtain the DDI datasets in three folds(note that for the scaffold split setting, the training dataset and the test dataset are the same across three fold).
The statistics about the data are as follows:
| --- | Molecules | DDIs | DDIE types |
| --- | --- | --- | --- |
| DDI dataset | 1,700 | 191,570 | 86 |
#### Drug-target interaction prediction
Since we used the DrugBAN framework for DTI prediction, the biosnap and BioSNAP datasets are downloaded in [DrugBAN datasets](https://github.com/peizhenbai/DrugBAN/tree/main/datasets). You can also find the data sources from [BindingDB](https://www.bindingdb.org/bind/index.jsp) and [BioSNAP](https://github.com/kexinhuang12345/MolTrans).
The statistics about the data are as follows:
| --- | Molecules | targets | DTIs |
| --- | --- | --- | --- |
| BindingDB | 14,643 | 2,623 | 49,199 |
| BioSNAP | 4,510 | 2,181 | 27,464 |

## Pretraining M2UMol
You can first check the 'settings' in 'run_pretrain.py', and modify them according to your needs. You can also set parameters directly in the training command, for example:

```
python run_pretrain.py --num_layers 3 --lr 0.001 --batch 32 --epochs 150 --tem 1.0 --mcls_loss_ratio 0.5 --output_name M2UMol
```

For the Text encoder, we utlized a pre-trained large language model (LLM) PubMedBERT proposed by Microsoft. We download pre-trained PubMedBERT at [this Hugging Face link](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/tree/main), and save them in a folder named 'pretrained-PubMedBERT'. For the 3D encoder, we utlized ComENet, a recently proposed method as the encoder, and implement it by using the [dive-into-graphs](https://github.com/divelab/DIG)

Note that the Text encoder and the 3D encoder are all further pre-trained in our pre-training phase, that is, the parameters of these models are not frozen. In addition, you can easily replace different large language models and 3D conformation encoders by modifying the parameters in 'layer.py'.

The paramters of our pre-trained M2UMol can be found in 'pre-trained_M2UMol.pt' in ... fold.

## Finetuning on three downstream tasks
We comprehensively verified the model performance of M2UMol through three downstream tasks: molecular property prediction, drug-drug interaction prediction and drug-target interaction prediction.

### Downstream task 1 :Molecular property prediction

### Downstream task 2 :Drug-drug interaction prediction
For the drug-drug interaction prediction, you can use the following commands for training and testing for three times with three different seeds (for the cold start split setting, it means three different split):
```
python run.py
```
You can also set different parameters in 'run.py' or run for one time by the following commands:
```
python main.py --seed 0 --lr 0.0005 --batch 256 --weight_decay 0.0002 --dropout 0.7 --split scaffold
```
where '--split cold' and '--split scaffold' can choose the split settings, we provide both cold start split setting and scaffold split setting, which we used in our paper.

### Downstream task 3 :Drug-target interaction prediction
For the drug-target interaction prediction, we utlized the framework of [DrugBAN](https://github.com/peizhenbai/DrugBAN) to evaluate the performances of M2UMol on the challenging cross-domain drug-target prediction tasks(named scaffold split setting in ou paper). Based on DrugBAN, we replace the molecular preprocessing procedure and the molecular encoder with our preprocessing procedure and M2UMol. You can use the following commands for training and testing for five times with five different seeds:
```
python run.py
```
You can also set different parameters in 'run.py' or run for one time by the following commands:
```
python main.py --cfg "configs/DrugBAN_DA.yaml" --data bindingdb --split "cluster" --seed 0
```
where '--cfg' is the config file of DrugBAN, for the scaffold split setting, we recommend using 'DrugBAN_DA.yaml'. '--data' can choose the DTI datasets, 'bindingdb' and 'biosnap' are available. '--split' can choose the split settings, 'random' and 'cluster' are available('cluster' denotes the scaffold split setting in our paper).


## Molecular analysis API of M2UMol
Considering that the proposed M2UMol has the ability to accurately focus on key molecular groups and perform cross-modal retrieval of multiple modes, we developed a molecular analysis API. It can be used as an AI-assisted drug design tool to visualize the importance of each part of a molecule, retrieve data from the library for four modes, and synthesize drugs that may be similar to it, while only inputting molecular SMILES. This information may be able to provide reference for researchers and guide the direction of experiments to a certain extent, which will help the process of drug discovery and drug development. To run the Molecular analysis tool, you can use the following command: 
```
cd molecular_analysis
python moleculartool.py
```
You will then be asked to enter the options. Here in, we take a molecule that did not appear in our pre-training as an example. For this molecule, only its SMILES is available in DrugBank, which can simulate an extreme case of molecular analysis where the available information about the molecule is sometimes very poor or only SMILES is available:
```
Please enter the SMILES of the molecule:COC(=O)N1CCC[C@H](NS(C)(=O)=O)[C@@H]1CO[C@H]1CC[C@H](CC1)C1=CC=CC=C1
Please enter the threshold of the attention coefficient:(-1,1), separated by Spaces:-1 0 0.5
Select whether Generic Name is required: yes or no:no
```
Note that some molecules may have long generic names, you can often choose 'no' to make the result more concise.

Then a html fils will be generated in 'HTMLpainter' fold, which named in the format "$YOUR SMILES$-molecular_information.html". You can open it up and view the molecular analysis we provided. Note that a folder named "$YOUR SMILES$" containing the attention visualizations and the molecular structure images will be generated, please keep this folder in the same directory as the html file). Here, the analysis results of the example molecules are：
<p align="center">
  <img src="pics/example_molecular_analysis.png" width="80%"/> 
</p>

The specific explanations for each section are as follows:
<p align="center">
  <img src="pics/tool-1.png" width="80%"/> 
</p>
<p align="center">
  <img src="pics/tool-2.png" width="80%"/> 
</p>
<p align="center">
  <img src="pics/tool-3.png" width="80%"/> 
</p>
<p align="center">
  <img src="pics/tool-4.png" width="80%"/> 
</p>



## How to use M2UMol in your own project
For how to use M2UMol as a general molecular encoder, we present an example in [here] and you can use it by using the command 'python toysample_encoder.py'. It can directly take SMILES as inputs and can learn fix representations with multimodal knowledge, which can be used as the feature or fingerprint of the molecule:
```python
import torch
from data_process import create_all_graph_data,construct_graph
from M2UMol import M2UMolencoder, model_load

# Input the SMILES
smiles_list=['ClC1=CC2=C(NC(=O)CN=C2C2=CC=CC=C2Cl)C=C1']
graph=construct_graph(create_all_graph_data(smiles_list),0).cuda()

#Define M2UMol encoder and load the pre-trained M2UMol model
model=M2UMolencoder().cuda()
model.eval()
model=model_load(model,'pre-trained_M2UMol.pt')

representation2d,generated_3d,generated_text,generated_bio=model(graph)
print('final representation')
print(representation2d.cpu().detach().numpy())
```
The output is:
```python
final representation
[[-1.20773971e+00  9.97009516e-01  5.55726171e-01  2.88000011e+00 ... -6.52980864e-01 -8.60271811e-01  1.30764246e-01  2.13101649e+00]]
```

If you want fine-tuning M2UMol in your own downstream tasks, we also present an example in [here]. Our pre-trained M2UMol can be easily used as a molecular encoder for a various molecular-related tasks, and because it is only a part of our pre-trained model, it is very efficient and lightweight for fine-tuning
```python
import torch
from data_process import create_all_graph_data,construct_graph,construct_graph_batch
from M2UMol import create_model, model_load
import numpy as np
from sklearn.metrics import roc_auc_score

#Prepare the input data and label. We recommend using the dataloader of pytorch or pytorch-geometric, here is just a simple example
smiles_list=['O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C',
             'Fc1cc(cc(F)c1)C[C@H](NC(=O)[C@@H](N1CC[C@](NC(=O)C)(CC(C)C)C1=O)CCc1ccccc1)[C@H](O)[C@@H]1[NH2+]C[C@H](OCCC)C1',
             'S1(=O)(=O)N(c2cc(cc3c2n(cc3CC)CC1)C(=O)N[C@H]([C@H](O)C[NH2+]Cc1cc(OC)ccc1)Cc1ccccc1)C',
             'S1(=O)(=O)C[C@@H](Cc2cc(O[C@H](COCC)C(F)(F)F)c(N)c(F)c2)[C@H](O)[C@@H]([NH2+]Cc2cc(ccc2)C(C)(C)C)C1',
             'S1(=O)(=O)N(c2cc(cc3c2n(cc3CC)CC1)C(=O)N[C@H]([C@H](O)C[NH2+]Cc1cc(ccc1)C(F)(F)F)Cc1ccccc1)C',
             'S(=O)(=O)(C(CCC)CCC)C[C@@H](NC(OCc1ccccc1)=O)C(=O)N[C@H]([C@H](O)C[NH2+]Cc1cc(OC)ccc1)Cc1ccccc1',
             'Fc1cc(cc(F)c1)C[C@H](NC(=O)c1cc(cc(c1)C)C(=O)N(CCC)CCC)[C@H](O)[C@@H]1[NH2+]CC[N@@H+](C1)Cc1ccccc1',
             'S(=O)(=O)(N[C@@H]1C[C@H](C[C@@H](C1)C(=O)N[C@H]([C@@H](O)CC(=O)N[C@@H](CC(C)C)C(=O)N[C@H](C(=O)N([C@H](CC1CCCCC1)C(=O)N1CCC[C@H]1C(OC)=O)C)C)CC1CCCCC1)C(=O)N[C@H](C)C1CCCCC1)C',
             'S1(=O)(=O)C[C@@H](Cc2cc(F)c3NCC4(CCC(F)(F)CC4)c3c2)[C@H](O)[C@@H]([NH2+]Cc2cc(ccc2)C(C)(C)C)C1',
             'O=C(N1CC[C@H](C[C@H]1c1ccccc1)c1ccccc1)[C@@H]1C[NH2+]C[C@]12CCCc1c2cccc1']
all_label=[1,1,1,1,1,0,0,0,0,0]

#construct the graph data, the M2UMol model and the optimizer
graph=construct_graph_batch(create_all_graph_data(smiles_list)).cuda()
model,optimizer=create_model()
model.cuda()
model.eval()
model=model_load(model,'pre-trained_M2UMol.pt')
loss_fct = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):
    print('-------- Epoch ' + str(epoch + 1) + ' --------')

    model.train()
    label=all_label
    label = torch.from_numpy(np.array(label)).cuda()
    output = model(graph).squeeze()
    loss = loss_fct(output, label.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    label_ids = label.to('cpu').numpy()
    acc = roc_auc_score(label_ids.flatten().tolist(), output.flatten().tolist())
    print(acc)
```

## Cite Us
Feel free to cite this work if you find it useful to you!
```
@article{M2UMol,
    title={Multi-to-uni Modal Knowledge Transfer Pre-training for Molecular Representation Learning},
    author={Zhankun Xiong, Ziyan Wang, Feng Huang, Minyao Qiu, Shuyan Fang, Liuqing Yang, Ping Zhang, and Wen Zhang},
    title={Multi-to-uni Modal Knowledge Transfer Pre-training for Molecular Representation Learning},
    year={2024},
}
```
