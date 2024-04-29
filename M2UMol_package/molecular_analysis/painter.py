

def paint_attentionmap(smiles,attentionmap,name,threshold=None):
    from rdkit import Chem
    from rdkit.Chem import Draw
    import numpy as np
    import torch
    from rdkit_heatmaps import mapvalues2mol
    from rdkit_heatmaps.utils import transform2png
    smiles=smiles
    test_mol = Chem.MolFromSmiles(smiles)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in test_mol.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats
    a=np.zeros((attentionmap.shape[0],attentionmap.shape[0]))

    for i in range(edge_list.shape[0]):
        a[edge_list[i][0],edge_list[i][1]]=1
        a[edge_list[i][1], edge_list[i][0]] = 1

    afinal=attentionmap*a
    atom_weights=np.sum(afinal,axis=1)
    bond=[]
    for i in range(edge_list.shape[0]//2):
        bond.append(afinal[edge_list[i][0],edge_list[i][1]])
    bond=np.array(bond)
    atom_weights=atom_weights
    bond_weights=bond
    all=np.concatenate((atom_weights,bond_weights),0)
    atom_weights = atom_weights - np.mean(atom_weights)
    atom_weights = atom_weights / np.max(np.abs(atom_weights))
    bond_weights = bond_weights - np.mean(bond_weights)
    bond_weights = bond_weights / np.max(np.abs(bond_weights))
    test_mol = Draw.PrepareMolForDrawing(test_mol)
    if threshold is not None:
        atom_weights[atom_weights<threshold]=0
        bond_weights[bond_weights<threshold]=0
    canvas = mapvalues2mol(test_mol, atom_weights,bond_weights)
    img = transform2png(canvas.GetDrawingText())
    img.save(str(name)+'.jpg')
    
def paint_2d_molecule(mol,name,path):
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DCairo
    from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    test_mol = mol
    d = rdMolDraw2D.MolDraw2DCairo(800, 400)
    do = d.drawOptions()
    do.padding = 0.2
    do.bondLineWidth = 5
    d.SetDrawOptions(do)
    tmp = rdMolDraw2D.PrepareMolForDrawing(test_mol)
    d.DrawMolecule(test_mol)
    d.FinishDrawing()
    d.WriteDrawingText(path+'/'+str(name)+'.png')
    #img.save(str(name)+'.jpg')
    
def paint_3d_molecule(mol,name,path):   
    from rdkit.Chem import Draw
    image=Draw.MolToImage(mol, size=(800,400))
    image.save(path+'/'+str(name)+'.png')