import torch
from data_process import create_all_graph_data,construct_graph
from M2UMol import M2UMolencoder, model_load

smiles_list=['ClC1=CC2=C(NC(=O)CN=C2C2=CC=CC=C2Cl)C=C1']
graph=construct_graph(create_all_graph_data(smiles_list),0).cuda()

model=M2UMolencoder().cuda()
model.eval()
model=model_load(model,'pre-trained_M2UMol.pt')

representation2d,generated_3d,generated_text,generated_bio=model(graph)
print('final representation')
print(representation2d.cpu().detach().numpy())
print('generated 3D representation')
print(generated_3d.cpu().detach().numpy())
print('generated Text representation')
print(generated_text.cpu().detach().numpy())
print('generated 3D representation')
print(generated_bio.cpu().detach().numpy())

