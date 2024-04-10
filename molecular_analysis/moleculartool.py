import numpy as np
from data_process import create_all_graph_data,construct_graph
from M2UMol import M2UMolencoder
import painter as paint
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
import HTMLpainter.htmlpainter as htmlp
import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1, deterministic=False)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--genericname', type=str, default='no', help='whether Generic Name is required')
args = parser.parse_args()


maskmat=np.load('drugbank_multimodaldata/mask.npy')

smilesinput=input("Please enter the SMILES of the molecule: ")
att_input = input("Please enter the threshold of the attention coefficient:(-1,1), separated by Spaces: ")
thresholdinput = att_input.split(' ')
thresholdinput = [float(item.strip()) for item in thresholdinput if item.strip()]
if args.genericname=='no':
    showflag=0
if args.genericname=='yes':
    showflag=1

    
smiles_list=[str(smilesinput)]
#smiles_list=['CC(=O)OC1=CC=CC=C1C(O)=O']
# print(smiles_list)
# print(smiles_list1)
mat=create_all_graph_data(smiles_list)
graph=construct_graph(mat,0)

model=M2UMolencoder()


model_dict = model.state_dict()
pretrained_dict = torch.load('pre-trained_M2UMol.pt')#,map_location=torch.device('cpu'))  #
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)  
model.to(device)
graph=graph.to(device)
model.eval()
representation2d,generated_3d,generated_text,generated_bio=model(graph)

########     Paint attention map
attentionmap = model.forward_view(graph)
attentionmap=attentionmap.cpu().detach().numpy()

########    Multimodal retrieval
import pickle
import csv
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from rdkit import Chem
drug_list=[]
with open(r'drugbank_multimodalrep/have2ddurglist.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            drug_list.append(row[0])
def dict2mat(dicts):
    druglist=[]
    mat=[]
    for i in range(len(drug_list)):
        try:
            mat.append(dicts[str(drug_list[i])])
            druglist.append(drug_list[i])
        except:
            continue
    mat=np.array(mat).squeeze()
    return mat,druglist
with open(r"drugbank_multimodalrep/mulallpre_fitune_2d0.pkl", "rb") as file: 
    mol2d = pickle.load(file)
with open(r"drugbank_multimodalrep/mulallpre_fitune_3d0.pkl", "rb") as file: 
    mol3d = pickle.load(file)
with open(r"drugbank_multimodalrep/mulallpre_fitune_wenben0.pkl", "rb") as file:
    moltext = pickle.load(file)
with open(r"drugbank_multimodalrep/mulallpre_fitune_finger0.pkl", "rb") as file:
    molbio = pickle.load(file)
    
mat2d,druglist=dict2mat(mol2d)
mat3d,_=dict2mat(mol3d)
mattext,_=dict2mat(moltext)
matbio,_=dict2mat(molbio)
from scipy.spatial.distance import pdist, squareform
representation2d=representation2d.cpu().detach().numpy()
generated_3d=generated_3d.cpu().detach().numpy().reshape(1,-1)
generated_text=generated_text.cpu().detach().numpy().reshape(1,-1)
generated_bio=generated_bio.cpu().detach().numpy().reshape(1,-1)
path = "./HTMLpainter/" + str(smiles_list[0])
import os
os.makedirs(path, exist_ok=True)
paint.paint_attentionmap(smiles_list[0],attentionmap,path+'/attention1',threshold=thresholdinput[0])  #threshold represents the range of attention values that you want to display. Areas below threshold are not highlighted. If there are molecular-related problems during the drawing process, you can try to draw using Canonical SMILES
paint.paint_attentionmap(smiles_list[0],attentionmap,path+'/attention2',threshold=thresholdinput[1])
paint.paint_attentionmap(smiles_list[0],attentionmap,path+'/attention3',threshold=thresholdinput[2])



smiles=smiles_list[0]
threshold=[]
d2list=[]
d3list=[]
textlist=[]
textdlist=[]
biolist=[]
biotextlist=[]
alllist=[]
threshold=thresholdinput

sdfs_2d = Chem.SDMolSupplier("drugbank_multimodaldata/2Dstructures.sdf",removeHs=False)
molecules2d={}
molecules2dname={}
for mol in sdfs_2d:
    if mol is not None:
        molecules2d[mol.GetProp('DRUGBANK_ID')]=mol
        molecules2dname[mol.GetProp('DRUGBANK_ID')]=mol.GetProp('GENERIC_NAME')


print('-------------------------------    2D modality')
distances = pdist(np.vstack((representation2d, mat2d)), 'euclidean')
distance_matrix = squareform(distances)
row, col = np.diag_indices_from(distance_matrix)
distance_matrix[row,col] = 10000000000
sim2d=distance_matrix[0]
sorted_indices = np.argsort(distance_matrix[0])
topsimdrug_list=[druglist[sorted_indices[0]-1],druglist[sorted_indices[1]-1],druglist[sorted_indices[2]-1],druglist[sorted_indices[3]-1],druglist[sorted_indices[4]-1]]
topsimdrug=''
for i in range(5):
    topsimdrug= str(topsimdrug+topsimdrug_list[i])+'    '
    if showflag == 0:
        d2list.append(str(topsimdrug_list[i]))
    if showflag == 1:
        d2list.append(str(topsimdrug_list[i]) + ":" + str(molecules2dname[topsimdrug_list[i]]))
print('top-5 drugs in 2D modality: '+str(topsimdrug))

for i in range(5):
    if showflag==0:
        d2list.append(str(topsimdrug_list[i]))
    if showflag==1:
        d2list.append(str(topsimdrug_list[i])+":"+str(molecules2dname[topsimdrug_list[i]]))
    #paint.paint_2d_molecule(molecules2d[topsimdrug_list[i]],'2D-top-'+str(i)+'-'+str(topsimdrug_list[i])+'-'+str(molecules2dname[topsimdrug_list[i]]),path)
    paint.paint_2d_molecule(molecules2d[topsimdrug_list[i]],'2D-top-' + str(i), path)
    
print('-------------------------------    3D modality')
distances = pdist(np.vstack((generated_3d, mat3d)), 'euclidean')
distance_matrix = squareform(distances)
row, col = np.diag_indices_from(distance_matrix)
distance_matrix[row,col] = 10000000000
sim3d=distance_matrix[0]
sorted_indices = np.argsort(distance_matrix[0])
topsimdrug_list=[druglist[sorted_indices[0]-1],druglist[sorted_indices[1]-1],druglist[sorted_indices[2]-1],druglist[sorted_indices[3]-1],druglist[sorted_indices[4]-1]]
topsimdrug=''
for i in range(5):
    topsimdrug= str(topsimdrug+topsimdrug_list[i])+'    '
    if showflag == 0:
        d3list.append(topsimdrug_list[i])
    if showflag == 1:
        d3list.append(topsimdrug_list[i] + ":" + str(molecules2dname[topsimdrug_list[i]]))
print('top-5 drugs in 3D modality: '+str(topsimdrug))

all_drug_list=[]
with open(r'drugbank_multimodaldata/drug_list.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            all_drug_list.append(row[0])
maskmat=np.load('drugbank_multimodaldata/mask.npy')
topsim3ddrug=[]
flag=0
for k in range(sorted_indices.shape[0]):
    if flag==5:
        break
    if int(maskmat[all_drug_list.index(druglist[sorted_indices[k]-1])][3])==1:
        topsim3ddrug.append(druglist[sorted_indices[k]-1])
        flag=flag+1

sdfs_3d = Chem.SDMolSupplier("drugbank_multimodaldata/3Dstructures.sdf",removeHs=False)
molecules3d={}
molecules3dname={}
kk=0
for mol in sdfs_3d:
    if mol is not None:
        kk=kk+1
        molecules3d[mol.GetProp('DRUGBANK_ID')]=mol#.GetProp('SMILES')
        molecules3dname[mol.GetProp('DRUGBANK_ID')]=mol.GetProp('GENERIC_NAME')
for i in range(5):
    print(molecules3d[topsim3ddrug[i]])
    if showflag == 0:
        d3list.append(str(topsim3ddrug[i]))
    if showflag == 1:
        d3list.append(str(topsim3ddrug[i]) + ":" + str(molecules2dname[topsim3ddrug[i]]))
    paint.paint_3d_molecule(molecules3d[topsim3ddrug[i]],'3D-top-'+str(i),path)
    
print('-------------------------------    Text modality')
distances = pdist(np.vstack((generated_text, mattext)), 'euclidean')
distance_matrix = squareform(distances)
row, col = np.diag_indices_from(distance_matrix)
distance_matrix[row,col] = 10000000000
simtext=distance_matrix[0]
sorted_indices = np.argsort(distance_matrix[0])
topsimdrug_list=[druglist[sorted_indices[0]-1],druglist[sorted_indices[1]-1],druglist[sorted_indices[2]-1],druglist[sorted_indices[3]-1],druglist[sorted_indices[4]-1]]
topsimdrug=''
for i in range(5):
    topsimdrug= str(topsimdrug+topsimdrug_list[i])+'    '
    if showflag == 0:
        textdlist.append(topsimdrug_list[i])
    if showflag == 1:
        textdlist.append(topsimdrug_list[i] + ":" + str(molecules2dname[topsimdrug_list[i]]))
print('top-5 drugs in Text modality: '+str(topsimdrug))

all_drug_list=[]
with open(r'drugbank_multimodaldata/drug_list.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            all_drug_list.append(row[0])
maskmat=np.load('drugbank_multimodaldata/mask.npy')
topsimtextdrug=[]
flag=0
for k in range(sorted_indices.shape[0]):
    if flag==5:
        break
    if int(maskmat[all_drug_list.index(druglist[sorted_indices[k]-1])][0])==1:
        topsimtextdrug.append(druglist[sorted_indices[k]-1])
        flag=flag+1

all_drugdescription=[]
with open(r'drugbank_multimodaldata/description-sup.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            all_drugdescription.append(row[0])
for i in range(5):
    if showflag == 0:
        textdlist.append(topsimtextdrug[i])
    if showflag == 1:
        textdlist.append(topsimtextdrug[i] + ":" + str(molecules2dname[topsimtextdrug[i]]))
    print(str(topsimtextdrug[i])+'    '+str(all_drugdescription[all_drug_list.index(topsimtextdrug[i])]))
    textlist.append(str(all_drugdescription[all_drug_list.index(topsimtextdrug[i])]))
    
    
print('-------------------------------    Bio modality')
distances = pdist(np.vstack((generated_bio, matbio)), 'euclidean')
distance_matrix = squareform(distances)
row, col = np.diag_indices_from(distance_matrix)
distance_matrix[row,col] = 10000000000
simbio=distance_matrix[0]
sorted_indices = np.argsort(distance_matrix[0])
topsimdrug_list=[druglist[sorted_indices[0]-1],druglist[sorted_indices[1]-1],druglist[sorted_indices[2]-1],druglist[sorted_indices[3]-1],druglist[sorted_indices[4]-1]]
topsimdrug=''
for i in range(5):
    topsimdrug= str(topsimdrug+topsimdrug_list[i])+'    '
    if showflag == 0:
        biolist.append(topsimdrug_list[i])
    if showflag == 1:
        biolist.append(topsimdrug_list[i] + ":" + str(molecules2dname[topsimdrug_list[i]]))
print('top-5 drugs in Bio modality: '+str(topsimdrug))

all_drug_list=[]
with open(r'drugbank_multimodaldata/drug_list.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            all_drug_list.append(row[0])
maskmat=np.load('drugbank_multimodaldata/mask.npy')
topsimbiodrug=[]
flag=0
for k in range(sorted_indices.shape[0]):
    if flag==5:
        break
    if int(maskmat[all_drug_list.index(druglist[sorted_indices[k]-1])][4])==1 or int(maskmat[all_drug_list.index(druglist[sorted_indices[k]-1])][5])==1 or int(maskmat[all_drug_list.index(druglist[sorted_indices[k]-1])][6])==1:
        topsimbiodrug.append(druglist[sorted_indices[k]-1])

target=np.load('drugbank_multimodaldata/targetfea.npy')
enzyme=np.load('drugbank_multimodaldata/enzymefea.npy')
cate=np.load('drugbank_multimodaldata/drugcatefea.npy')

targetlist=[]
with open(r'drugbank_multimodaldata/target_list.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            targetlist.append(row[0])
enzymelist=[]
with open(r'drugbank_multimodaldata/enzyme_list.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            enzymelist.append(row[0])
catelist=[]
with open(r'drugbank_multimodaldata/drugcate_list.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            catelist.append(row[0])

all_drugdescription=[]
with open(r'drugbank_multimodaldata/description-sup.csv', 'r') as f: #_xiao   #drug_list_xiao
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            all_drugdescription.append(row[0])
for i in range(5):
    if showflag == 0:
        biolist.append(topsimbiodrug[i])
    if showflag == 1:
        biolist.append(topsimbiodrug[i] + ":" + str(molecules2dname[topsimbiodrug[i]]))
    bioinfo=''
    k=np.array(target[all_drug_list.index(topsimbiodrug[i])]).astype("bool")
    targetinfo=np.array(targetlist)[k].tolist()
    if str(targetinfo)=='[]':
        targetinfo='Unknown'
    targetinfo='target: '+str(targetinfo).replace('[','').replace(']','').replace("'",'').replace(",",'|')
    k=np.array(enzyme[all_drug_list.index(topsimbiodrug[i])]).astype("bool")
    enzymeinfo=np.array(enzymelist)[k].tolist() 
    if str(enzymeinfo)=='[]':
        enzymeinfo='Unknown'
    enzymeinfo='enzyme: '+str(enzymeinfo).replace('[','').replace(']','').replace("'",'').replace(",",'|')
    k=np.array(cate[all_drug_list.index(topsimbiodrug[i])]).astype("bool")
    cateinfo=np.array(catelist)[k].tolist()  
    if str(cateinfo)=='[]':
        cateinfostring='Unknown'
    if str(cateinfo)!='[]':
        cateinfostring = ''
        for cateindex in range(len(cateinfo)):
            cateinfostring=cateinfostring+','+cateinfo[cateindex].replace(",",'$')
    cateinfo='drug categories: '+str(cateinfostring).replace('[','').replace(']','').replace("'",'').replace(",",'|').replace("$",',')

    bioinfo=str(targetinfo+'        '+enzymeinfo+'        '+cateinfo)
    print(str(topsimbiodrug[i])+'\n'+bioinfo+'\n'+'\n')
    biotextlist.append(bioinfo)
    
print('-------------------------------    The most similar drugs are in a comprehensive perspective')
simall=(sim2d+sim3d+simtext+simbio)/4
sorted_indices = np.argsort(distance_matrix[0])
topsimdrug_list=[druglist[sorted_indices[0]-1],druglist[sorted_indices[1]-1],druglist[sorted_indices[2]-1],druglist[sorted_indices[3]-1],druglist[sorted_indices[4]-1]]
topsimdrug=''
for i in range(5):
    topsimdrug= str(topsimdrug+topsimdrug_list[i])+'    '
    if showflag == 0:
        alllist.append(topsimdrug_list[i])
    if showflag == 1:
        alllist.append(topsimdrug_list[i] + ":" + str(molecules2dname[topsimdrug_list[i]]))
print('top-5 most similar drugs in a comprehensive perspective: '+str(topsimdrug))

htmlp.htmlpainter(smiles,threshold,d2list,d3list,textlist,textdlist,biolist,biotextlist,alllist)