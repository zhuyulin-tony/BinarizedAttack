import torch
import torch.nn as nn

class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

myroundfunction = my_round_func.apply

class multiple_AS(nn.Module):
    def __init__(self, target_lst, atk_link_idx, n_node, device):
        super().__init__()
        self.lst = target_lst
        self.n = n_node
        self.atk_link_idx = atk_link_idx
        self.device = device
        self.par = nn.Parameter(0.49 * torch.ones((int(0.5*self.n*(self.n-1)),1))[self.atk_link_idx].to(self.device))
    
    def get_Z(self):
        Z_continue = 0.49 * torch.ones((int(0.5*self.n*(self.n-1)),1)).to(self.device)
        Z_continue[self.atk_link_idx] = self.par
        return Z_continue
        
    def perturb(self, tri):
        Z_continue = self.get_Z()
        Z = -2 * myroundfunction(Z_continue) + 1
        tri_perturb = tri.clone()
        tri_perturb[:,2:] = (tri[:,2:]-0.5) * Z + 0.5
        return tri_perturb
    
    def adjacency_matrix(self, tri):
        A = torch.sparse_coo_tensor(tri[:,:2].T, tri[:,2], size=[self.n,self.n]).to_dense()
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def filter_potential_singletons(self, tri):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """
        modified_adj = self.adjacency_matrix(tri)
        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        
        idx = torch.triu_indices(self.n, self.n, offset=1)
        return flat_mask[idx[0],idx[1]].reshape(-1,1)
    
    def sparse_matrix_power(self, A):
        A_sp = A.to_sparse()
        A_sp = torch.sparse_coo_tensor(A_sp.indices(), A_sp.values(), size=[self.n,self.n])
        return torch.sparse.mm(torch.sparse.mm(A_sp, A_sp), A_sp).to_dense()
        
    def extract_NE(self, A):
        N = torch.sum(A, 1)
        E = torch.sum(A, 1) + 0.5 * torch.diag(self.sparse_matrix_power(A)).T
        N = N.reshape(-1,1)
        E = E.reshape(-1,)
        return N, E
    
    def OLS_estimation(self, N, E):
        logN = torch.log(N + 1e-20)
        logE = torch.log(E + 1e-20)
        logN1 = torch.cat((torch.ones((len(logN),1)).to(self.device), logN), 1)
        return torch.inverse(logN1.T @ logN1) @ logN1.T @ logE
       
    def forward(self, tri):
        tri_perturb = self.perturb(tri)
        A = self.adjacency_matrix(tri_perturb)
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0]
        w = theta[1]
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.exp(b) * (N[self.lst[i]]**w) - E[self.lst[i]])**2
        return tmp
    
    def true_AS(self, tri):
        tri_perturb = self.perturb(tri)
        A = self.adjacency_matrix(tri_perturb)
        N, E = self.extract_NE(A)
        theta = self.OLS_estimation(N, E)
        b = theta[0]
        w = theta[1] 
        tmp = 0.
        for i in range(len(self.lst)):
            tmp += (torch.max(E[self.lst[i]], torch.exp(b)*(N[self.lst[i]]**w))/\
                    torch.min(E[self.lst[i]], torch.exp(b)*(N[self.lst[i]]**w)))*\
                    torch.log(torch.abs(E[self.lst[i]]-torch.exp(b)*(N[self.lst[i]]**w))+1)
        return tmp