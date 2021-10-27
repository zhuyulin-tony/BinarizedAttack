import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from shutil import copyfile

parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--dataset', type=str, default='ca_grqc', choices=['ca_grqc'], help='dataset')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--size', type=int, default=10, choices=[10,30], help='target nodes size')
parser.add_argument('--B', type=int, default=300, choices=[40,50,100,120,200], help='budget B')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu'], help='device')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

root_dir = os.getcwd().replace('\\', '/')

A = np.loadtxt(root_dir + '/dataset/'+args.dataset+'/adj.txt').astype('float')
n = len(A)

triple = []
for i in range(n):
    for j in range(i+1,n):
        triple.append([i,j,A[i,j]])
triple = np.array(triple)

target_node_lst = np.loadtxt(root_dir + '/dataset/'+args.dataset+'/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '.txt').astype('int')

model_dir = root_dir + '/dataset/'+args.dataset+'/BinarizedAttackd/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '/saved_ckt'
save_dir = root_dir + '/dataset/'+args.dataset+'/BinarizedAttackd/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '/sort_ckt'
try:
    os.makedirs(save_dir)
except:
    pass
# [1:] remove loss.txt file.
#model_name_lst = os.listdir(model_dir)[1:]
model_name_lst = os.listdir(model_dir)

def Pick_ckt(lst):
    table = np.zeros((len(lst), 2))
    for i in range(len(lst)):
        table[i,0] = int(re.findall(r'\d+', lst[i])[0])
        table[i,1] = float(re.findall(r'\d+', lst[i])[1] + '.' \
                         + re.findall(r'\d+', lst[i])[2])
        print(i, table[i,0], table[i,1])
    B_lst = list(set(table[:,0]))
    tmp = np.zeros((len(B_lst), 3))
    for j in range(len(B_lst)):
        tmp[j,0] = B_lst[j]
        tmp[j,1] = np.where(table[:,0] == B_lst[j])[0]\
                            [np.argmin(table[np.where(table[:,0] == B_lst[j])][:,1])]
        tmp[j,2] = np.min(table[np.where(table[:,0] == B_lst[j])][:,1])
        
    return tmp

ckt = Pick_ckt(model_name_lst)

for i in range(len(ckt)):
    idx = int(ckt[i,1])
    copyfile(model_dir + '/' + model_name_lst[idx], save_dir + '/' + model_name_lst[idx])

plt.plot(ckt[:,2])

np.savetxt(root_dir + '/dataset/'+args.dataset+ '/BinarizedAttackd/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '/AS.txt', ckt[:,2])