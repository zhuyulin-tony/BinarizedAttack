import argparse
import os
import torch
import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['bitcoin_alpha','ca_grqc','wikivote','cora_ml','citeseer','blogcatalog'], help='dataset')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--size', type=int, default=10, choices=[10,30], help='target nodes size')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
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

model_dir = root_dir + '/dataset/'+args.dataset+'/ContinuousA/' + 'rand'+ str(args.size) + '_' + str(args.trial) + '/saved_ckt'
try:
    os.makedirs(model_dir)
except:
    pass

class multiple_AS(nn.Module):
    def __init__(self, target_lst, n_node, tri):
        super().__init__()
        self.lst = target_lst
        self.n = n_node
        self.edge = nn.Parameter(torch.Tensor(tri[:,2]).to(args.device))
        self.tri = tri
    
    def adjacency_matrix(self):
        A = torch.sparse_coo_tensor(self.tri[:,:2].T, self.edge, size=[self.n,self.n]).to_dense()
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def extract_NE(self, A):
        N = torch.sum(A, 1)
        E = torch.sum(A, 1) + 0.5 * torch.diag(torch.matrix_power(A, 3)).T
        N = N.reshape(-1,1)
        E = E.reshape(-1,)
        return N, E
    
    def OLS_estimation(self, N, E):
        logN = torch.log(N)
        logE = torch.log(E)
        logN1 = torch.cat((torch.ones((len(logN),1)).to(args.device), logN), 1)
        return torch.inverse(logN1.T @ logN1) @ logN1.T @ logE
        
    def forward(self):
        A = self.adjacency_matrix()
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0]
        w = theta[1]
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.exp(b) * (N[self.lst[i]]**w) - E[self.lst[i]])**2
        return tmp
    
    def true_AS(self):
        A = self.adjacency_matrix()
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0]
        w = theta[1] 
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.max(E[self.lst[i]],torch.exp(b)*(N[self.lst[i]]**w))\
                   /torch.min(E[self.lst[i]],torch.exp(b)*(N[self.lst[i]]**w)))*\
                    torch.log(torch.abs(E[self.lst[i]]-torch.exp(b)*(N[self.lst[i]]**w))+1)
        return tmp

model = multiple_AS(target_lst = target_node_lst, n_node = n, tri = triple).to(args.device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) 
losses = []

for i in range(args.epoch):
    loss = model()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # projection gradient descent.
    model.edge.data.clamp_(1e-3, 1.-1e-3)
    
    AScore = model.true_AS()
    print('iters:', i, 'AS:', np.round(AScore.item(),3), 'loss:', np.round(loss.item(), 3))
    losses.append(loss.item())
np.savetxt(model_dir+'/loss.txt', losses)
torch.save(model.state_dict(),model_dir+"/ckt.pth")