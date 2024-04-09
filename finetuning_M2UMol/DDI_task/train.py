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
import csv
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

def train_model_ddi_2d(model, optimizer_ddi, train_loader, val_loader, test_loader,args):
    #print("model",model)
    optimizer=optimizer_ddi
    model_dict = model.state_dict()
    pretrained_dict = torch.load('run/pre-trained_M2UMol.pt')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    pretrained_dictname={k for k, v in pretrained_dict.items()}
    print('check loaded pre-trained layer')
    print(pretrained_dictname)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  ##
    print('load pre-trained M2UMol success')


    model.cuda()#.to(device)
    loss_fct = torch.nn.CrossEntropyLoss()

    if args.cuda:
        model.to(device)

    print('Start Training...')
    t_total = time.time()
    stoping = 0
    loss_history = []
    max_auc=0
    max_f1=0

    for epoch in range(args.epochs):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (batch) in enumerate(train_loader):
            model.train()

            alltwograph, alltwographt,label = batch
            alltwograph.to(device)
            alltwographt.to(device)
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.to(device)
            output= model(i, alltwograph,alltwographt)
            log = torch.squeeze(output)

            label.to(device)
            loss1 = loss_fct(log, label.long())
            loss_train =  loss1
            loss_history.append(loss_train.cpu().detach().numpy())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 200 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        y_pred_train1 = []
        y_label_train = np.array(y_label_train)
        y_pred_train = np.array(y_pred_train).reshape((-1, 86))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    y_pred_train1.append(j)
                    break
        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
        precision1 = precision_score(y_label_train, y_pred_train1, average='macro')

        print(acc)
        print(f1_score1)
        print(recall1)
        print(precision1)


        acc_val, f1_val, recall_val,precision_val, loss_val = test(model, val_loader, args,0)
        #if acc_val >= max_auc and f1_val>=max_f1
        if acc_val >= max_auc and f1_val>=max_f1:
            model_max = copy.deepcopy(model)
            max_auc = acc_val
            max_f1=f1_val
        print('epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'auroc_train: {:.4f}'.format(acc),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val),
              'f1_val: {:.4f}'.format(f1_val),
              'recall_val: {:.4f}'.format(recall_val),
              'precision_val: {:.4f}'.format(precision_val),
              'time: {:.4f}s'.format(time.time() - t))


        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test, f1_test, recall_test,precision_test, loss_test= test(model_max, test_loader, args,1)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test),'recall_test: {:.4f}'.format(recall_test))
    torch.save(model_max.state_dict(), 'M2UMolDDI'+'_'+str(args.seed)+'.pt')

    


def test(model, loader, args,printfou):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    model.eval()

    y_pred = []
    y_label = []
    seed=args.seed
    for i, (batch) in enumerate(loader):
        alltwograph, alltwographt, label = batch

        alltwograph.to(device)
        alltwographt.to(device)
        label = np.array(label, dtype=np.int64)
        label = torch.from_numpy(label)
        if args.cuda:
            label = label.to(device)


        output = model(i, alltwograph,alltwographt)
        log = torch.squeeze(m(output))

        loss1 = loss_fct(log, label.long())
        loss = loss1 #+ args.loss_ratio2 * loss2 + args.loss_ratio3 * loss3

        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + output.flatten().tolist()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()


    y_pred_train1=[]
    y_label_train = np.array(y_label)
    y_pred_train = np.array(y_pred).reshape((-1, 86))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    y_label_train1 = np.zeros((y_label_train.shape[0], 86))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    auc_hong=0
    aupr_hong=0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):
        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:
            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)
    print(auc1)
    print(aupr)
    print(auc_macro)
    print(aupr_macro)
    if printfou==1:

        with open(args.out_file, 'a') as f:
            f.write(str(seed)+'  '+str(acc)+'   '+str(f1_score1)+'   '+str(recall1)+'   '+str(precision1)+'   '+str(auc1)+'   '+str(aupr)+'   '+str(auc_macro)+'   '+str(aupr_macro)+'\n')

    return acc,f1_score1,recall1,precision1,loss