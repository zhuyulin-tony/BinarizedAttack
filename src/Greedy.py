import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='')
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['bitcoin_alpha','ca_grqc','wikivote','cora_ml','citeseer','blogcatalog'], help='dataset')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--size', type=int, default=10, choices=[10,30], help='target nodes size')
parser.add_argument('--B', type=int, default=35, choices=[35], help='budget B')
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

mod_dir = root_dir + '/dataset/'+args.dataset+'/GradMaxSearch/' + 'rand'+ str(args.size) + '_' + str(args.trial)

try:
    os.makedirs(mod_dir)
except:
    pass

class multiple_AS(nn.Module):
    def __init__(self, target_lst, n_node, device):
        super().__init__()
        self.lst = target_lst
        self.n = n_node
        self.device = device
    
    def adjacency_matrix(self, tri):
        A = torch.sparse_coo_tensor(tri[:,:2].T, tri[:,2], size=[self.n,self.n]).to_dense()
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def sparse_matrix_power(self, A, tau):
        A_sp = A.to_sparse()
        A_sp = torch.sparse_coo_tensor(A_sp.indices(), A_sp.values(), size=[self.n,self.n])
        return torch.sparse.mm(torch.sparse.mm(A_sp, A_sp), A_sp).to_dense()
    
    def extract_NE(self, A):
        N = torch.sum(A, 1)
        E = torch.sum(A, 1) + 0.5 * torch.diag(self.sparse_matrix_power(A, 3)).T
        N = N.reshape(-1,1)
        E = E.reshape(-1,)
        return N, E
    
    def OLS_estimation(self, N, E):
        logN = torch.log(N + 1e-20)
        logE = torch.log(E + 1e-20)
        logN1 = torch.cat((torch.ones((len(logN),1)).to(self.device), logN), 1)
        return torch.linalg.pinv(logN1) @ logE
        
    def forward(self, tri):
        A = self.adjacency_matrix(tri)
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0]
        w = theta[1]
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.exp(b) * (N[self.lst[i]]**w) - E[self.lst[i]])**2
        return tmp
    
    def true_AS(self, tri):
        A = self.adjacency_matrix(tri)
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

def Poison_attack(model, triple, B):
    triple_copy = triple.copy()
    np.savetxt(mod_dir+'/triple_mod_'+str(0)+'.txt',triple_copy,fmt='%d')
    triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True)
    AS = []
    perturb = []
    AS.append(model.true_AS(triple_torch).data.numpy()[0])
    print('initial AS:', model.true_AS(triple_torch).data.numpy()[0])
    
    for i in range(1,B+1):
        loss = model.forward(triple_torch)
        loss.backward()
        
        tmp = triple_torch.grad.data.numpy()
        grad = np.concatenate((triple_torch[:,:2].data.numpy(),tmp[:,2:]),1)
        
        v_grad = np.zeros((len(grad),3))
        for j in range(len(grad)):
            v_grad[j,0] = grad[j,0]
            v_grad[j,1] = grad[j,1]
            if triple_copy[j,2] == 0 and grad[j,2] < 0:
                v_grad[j,2] = grad[j,2]
            elif triple_copy[j,2] == 1 and grad[j,2] > 0:
                v_grad[j,2] = grad[j,2]
            else:
                continue
        v_grad = v_grad[np.abs(v_grad[:,2]).argsort()]
        # attack w.r.t gradient information.
        K = -1
        while v_grad[K][:2].astype('int').tolist() in perturb:
            K -= 1
            
        # do not delete edge from singleton.
        while v_grad[int(K)][2] > 0 and \
             (model.adjacency_matrix(triple_torch).data.numpy()[int(v_grad[int(K)][0])].sum() <= 1 or \
              model.adjacency_matrix(triple_torch).data.numpy()[int(v_grad[int(K)][1]) ].sum() <= 1):
            K -= 1
        
        target_grad = v_grad[int(K)]
        #print(K, target_grad)
        target_index = np.where(np.all((triple[:,:2] == target_grad[:2]), axis = 1) == True)[0][0]
        triple_copy[target_index,2] -= np.sign(target_grad[2])
        np.savetxt(mod_dir+'/triple_mod_'+str(i)+'.txt',triple_copy,fmt='%d')
        triple_torch = Variable(torch.from_numpy(triple_copy), requires_grad = True)
        perturb.append([int(target_grad[0]),int(target_grad[1])])
        true_AScore = model.true_AS(triple_torch).data.numpy()[0]
        AS.append(true_AScore)
        print('iter', i, 'AS:', true_AScore)
    AS = np.array(AS)    
    return triple_torch.data.numpy(), AS

model = multiple_AS(target_lst = target_node_lst, n_node = n, device = 'cpu')
A_mod, AS = Poison_attack(model, triple, args.B)
np.savetxt(mod_dir+'/AS.txt',AS)
#plt.plot(AS)