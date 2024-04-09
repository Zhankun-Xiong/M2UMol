import torch
from data_process import create_all_graph_data,construct_graph,construct_graph_batch
from M2UMol import create_model, model_load
import numpy as np
from sklearn.metrics import roc_auc_score
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




