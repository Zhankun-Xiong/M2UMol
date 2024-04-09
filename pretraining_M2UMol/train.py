import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve,auc
import os
import random
import copy
import torch.nn as nn
import molepre
import numpy as np
from loss_ import NtXent
import time

from molepre import return2d3d,returntext,returnbio
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)

def train_model(model, optimizer,sch, train_loader, args):
    lossnt=NtXent(args.tem)
    if args.cuda:
        model.to(device)
    loss_fct = torch.nn.CrossEntropyLoss()
    lossmax=100000000000

    print('Start Training...')
    t = time.time()
    for epoch in range(args.epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        kk=0
        trainepochloss=0
        for i in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                twograph,threegraph=return2d3d(i[1])
                text2d,text=returntext(i[1])
                bio=returnbio(i[1])
                threegraph.to(device)
                twograph.to(device)
                text2d.to(device)
                z1, z2, z3, z4, z5, z6, z1cls, z2cls, z3cls = model(twograph, threegraph, text2d, text, bio)
                loss1 = lossnt(z1, z2)
                loss2 = lossnt(z3, z4)
                loss3 = lossnt(z5, z6)
                aa = torch.cat((z1cls, z2cls, z3cls), dim=0)
                label = []
                for kk1 in range(aa.shape[0]):
                    if kk1 < z1.shape[0]:
                        label.append(0)
                        continue
                    if kk1 >= z1.shape[0] and kk1 < z1.shape[0] + z3.shape[0]:
                        label.append(1)
                        continue
                    else:
                        label.append(2)
                label = torch.tensor(label)
                label = label.to(device)
                loss4 = loss_fct(aa, label)
                loss_train = loss1 + loss2 + loss3 + args.mcls_loss_ratio*loss4


                trainepochloss=trainepochloss+loss_train.cpu().detach().numpy()
                loss_train.backward()
                optimizer.step()
                # if hasattr(torch.cuda, 'empty_cache'):
                #     torch.cuda.empty_cache()

                if kk % 350 == 0:
                    print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(kk + 1) + '/ loss_train: ' + str(loss_train.cpu().detach().numpy()))
                kk=kk+1
                # if hasattr(torch.cuda, 'empty_cache'):
                #     torch.cuda.empty_cache()
        sch.step()
        print('Pre-training loss')
        print(trainepochloss)
        if trainepochloss <= lossmax:
            lossmax =trainepochloss
            model_max = copy.deepcopy(model)


        if hasattr(torch.cuda, 'empty_cache'):
             torch.cuda.empty_cache()
        print('time: {:.4f}s'.format(time.time() - t))
        if epoch%50==0:
            path = os.path.join('run/',str(args.output_name)+'_%d_%3f_%3f.pt' % (epoch, args.lr, args.weight_decay))
            torch.save(model_max.state_dict(), path)
            path = os.path.join('run/',str(args.output_name)+'M2UMol_%d_%3f_%3f.pt' % (epoch, args.lr, args.weight_decay))
            torch.save(model.state_dict(), path)

    path = os.path.join('run/',str(args.output_name)+'maxmodel_%d_%3f_%3f.pt' % (args.epochs, args.lr,args.weight_decay))
    torch.save(model_max.state_dict(), path)
    path = os.path.join('run/',str(args.output_name)+'finalmodel_%d_%3f_%3f.pt' % (args.epochs, args.lr,args.weight_decay))
    torch.save(model.state_dict(), path)
