import os
#
with open('results.txt', 'a') as f:
    f.write('DrugBan-bindingdb-cluster')
    f.write('\n')
for i in range(5):
    print(i)
    #os.system('python main.py --cfg "configs/DrugBAN_DA.yaml" --data biosnap --split "random" --seed '+str(i)) #--seed '+str(ii)
    os.system('python main.py --cfg "configs/DrugBAN_DA.yaml" --data bindingdb --split "cluster" --seed '+str(i)) #--seed '+str(ii)
with open('test.txt', 'a') as f:
    f.write('\n')