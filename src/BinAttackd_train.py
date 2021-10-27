import argparse
import os
import torch
import numpy as np
import src.BinAttackd_model as attack_model
from shutil import copyfile
import re
import shutil

parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--dataset', type=str, default='ca_grqc', choices=['ca_grqc'], help='dataset')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--size', type=int, default=10, choices=[10,30], help='target nodes size')
parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')
parser.add_argument('--epoch', type=int, default=3000, help='epoch')
parser.add_argument('--B', type=int, default=300, choices=[300], help='budget B')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='device')

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

def direct_attack(a_lst, tri):
    atk_set = []
    for a in a_lst:
        for i in range(len(tri)):
            if tri[i,0] == a or tri[i,1] == a:
                atk_set.append(i)
        print(a)
    return atk_set
        
atk_link_idx = direct_attack(target_node_lst, triple)

model_dir = root_dir + '/dataset/'+args.dataset+'/BinarizedAttackd/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '/saved_ckt'
save_dir = root_dir + '/dataset/'+args.dataset+'/BinarizedAttackd/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '/sort_ckt'

try:
    os.makedirs(model_dir)
except:
    pass

try:
    os.makedirs(save_dir)
except:
    pass

def Pick_ckt(lst):
    table = np.zeros((len(lst), 2))
    for i in range(len(lst)):
        table[i,0] = int(re.findall(r'\d+', lst[i])[0])
        table[i,1] = float(re.findall(r'\d+', lst[i])[1] + '.' \
                         + re.findall(r'\d+', lst[i])[2])
        
    B_lst = list(set(table[:,0]))
    tmp = np.zeros((len(B_lst), 3))
    for j in range(len(B_lst)):
        tmp[j,0] = B_lst[j]
        tmp[j,1] = np.where(table[:,0] == B_lst[j])[0]\
                            [np.argmin(table[np.where(table[:,0] == B_lst[j])][:,1])]
        tmp[j,2] = np.min(table[np.where(table[:,0] == B_lst[j])][:,1])
        
    return tmp

lam_lst = [0,0.1,0.3,0.5,0.7,0.9,
           1,2,3,4,5,6,7,8,9,
           10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,
           100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,
           500,510,520,530,540,550,560,570,580,590,600,610,620,630,640,650,660,670,680,690,700,710,720,730,740,750,760,770,780,790,800,810,820,830,840,850,860,870,880,890,
           900,910,920,930,940,950,960,970,980,990,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,
           3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000,12000,13000,14000,
           15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000]

for lam in lam_lst:
    model = attack_model.multiple_AS(target_node_lst, atk_link_idx, n, args.device).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) 
    triple_torch = torch.from_numpy(triple).to(args.device)
    losses = []
    for i in range(args.epoch):
        loss = model(triple_torch) + lam * torch.sum(torch.abs(model.par))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # projection gradient descent.
        model.par.data.clamp_(1e-3, 1.-1e-3)
    
        AScore = model.true_AS(triple_torch)
        if torch.isnan(AScore).int().sum() > 0:
            break
        perturb = (model.par >= 0.5).sum().item()
        print('iters:', i, 'lam:', lam, 'AS:', np.round(AScore.item(),1), 'loss:', np.round(loss.item(), 1), 
              'perturb:', perturb)
        losses.append(loss.item())
        if perturb <= args.B:
            torch.save(model.state_dict(),model_dir+"/P="\
                                          +str(perturb)+",AS="+str(np.round(AScore.item(),1))+".pth")
    if lam_lst.index(lam) // 3:
        model_name_lst = os.listdir(model_dir)
        ckt = Pick_ckt(model_name_lst)
        for k in range(len(ckt)):
            idx = int(ckt[k,1])
            copyfile(model_dir + '/' + model_name_lst[idx], save_dir + '/' + model_name_lst[idx])
        
        shutil.rmtree(model_dir)
        #os.rmdir(model_dir)
        os.rename(save_dir, model_dir)
        os.makedirs(save_dir)
    
    
#np.savetxt(model_dir+'/loss.txt', losses)
