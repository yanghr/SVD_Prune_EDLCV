import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np

class SVD_Linear(nn.Module):
    def __init__(self,input_dim ,output_dim, rank=None, bias = True, activation_fn = None):
        super(SVD_Linear, self).__init__()
        if activation_fn is not None:
            self.activation_fn = activation_fn()
        else:
            self.activation_fn = None
        if rank is not None and rank<min(input_dim,output_dim):
            r = rank
        else:
            r = min(input_dim,output_dim)
        self.U = nn.Parameter(torch.empty(input_dim,r))#.cuda()
        self.Singular = nn.Parameter(torch.empty(r))#.cuda()
        self.V = nn.Parameter(torch.empty(r,output_dim))#.cuda()
        self.register_parameter('U',self.U)
        self.register_parameter('V',self.V)
        self.register_parameter('Singular',self.Singular)
        torch.nn.init.xavier_normal_(self.U)
        torch.nn.init.xavier_normal_(self.V)
        torch.nn.init.uniform_(self.Singular)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))#.cuda()
            self.register_parameter('bias',self.bias)
            torch.nn.init.constant_(self.bias,0.0)
        

    
    def forward(self,x):
        y = x.matmul(self.U)
        y = y.matmul(self.Singular.abs().diag())
        y = y.matmul(self.V)
#        valid_idx = torch.arange(self.Singular.size(0))[self.Singular.abs()>self.sensitivity]
#        U = self.U[:,valid_idx].contiguous()
#        V = self.V[valid_idx,:]
#        Sigma = self.Singular[valid_idx]
#        y = x.matmul(U)
#        y = y.matmul(Sigma.abs().diag())
#        y = y.matmul(V)

        if self.bias is not None:
            y = y.add(self.bias)
        if self.activation_fn is not None:
            y = self.activation_fn(y)
        return y

    def prun(self,sensitivity):
        assert sensitivity <= 1.0 and sensitivity >= 0.0
        energy = self.Singular**2
        _, sorted_idx = torch.sort(energy, descending=True)
        sum_e = torch.sum(energy)
        valid = 0
        for i in range(len(sorted_idx)):
            if energy[sorted_idx[:(i+1)]].sum()/sum_e >= sensitivity:
                valid = i+1
                break
        valid_idx = sorted_idx[:valid]
        #valid_idx = torch.arange(self.Singular.size(0))[self.Singular.abs()>sensitivity]
        self.U.data = self.U[:,valid_idx]
        self.V.data = self.V[valid_idx,:]
        self.Singular.data = self.Singular[valid_idx]


class SVD_LSTMCell(nn.Module):
    def __init__(self,input_size,hidden_size, rank=None,bias = True):
        super(SVD_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = SVD_Linear(input_size+hidden_size,4*hidden_size,rank,bias,None)

    def forward(self,x,prev_state = None):
        batch_size = x.size(0)
        #print(x.size())
        if prev_state is None:
            prev_state = [
            torch.zeros(batch_size,self.hidden_size,device = next(self.parameters()).device),
            torch.zeros(batch_size,self.hidden_size,device = next(self.parameters()).device)
            ]
        prev_hidden, prev_cell = prev_state
        stacked_inputs = torch.cat((x, prev_hidden), 1)#x is batch_size x input_size,stacked_inputs is batch_size x input_size+hidden_size
        gates = self.linear(stacked_inputs)#gates is batch_size x 4*hidden_size

        # chunk across channel dimension
        in_gate, remember_gate, cell_gate, out_gate = gates.chunk(4, 1)#each gate is batch_size x hidden_size

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)


        return hidden, cell

    def get_U(self):
        return self.linear.U

    def get_V(self):
        return self.linear.V

    def get_Sigma(self):
        return self.linear.Singular

class SVD_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout, rank=None,bias = True):
        super(SVD_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = torch.nn.ModuleList()
        self.drop=dropout
        
        if rank is not None:
            ranks = rank[0:num_layers]
        else:
            ranks = [None]*num_layers

        if isinstance(hidden_size,list) or isinstance(hidden_size,tuple):
            self.cells.append(SVD_LSTMCell(input_size,hidden_size[0],ranks[0],bias))
            for i in range(num_layers-1):
                self.cells.append(SVD_LSTMCell(hidden_size[i],hidden_size[i+1],ranks[i+1],bias))
        else:
            self.cells.append(SVD_LSTMCell(input_size,hidden_size,ranks[0],bias))
            for i in range(num_layers-1):
                self.cells.append(SVD_LSTMCell(hidden_size,hidden_size,ranks[i+1],bias))
        
    def forward(self,x,hidden):
        internal_states = []
        outputs = []
        step = x.size(0)#x is seq x batch_size x input_size
        for s in range(step):
            y = x[s]
            for i,cell in enumerate(self.cells):
                if s==0:
                    y1,new_c = cell(y,hidden)
                    y = self.drop(y1)
                    new_c = self.drop(new_c)
                    internal_states.append((y,new_c))
                else:
                    #h = self.drop(internal_states[i])
                    y1,new_c = cell(y,internal_states[i])
                    y = self.drop(y1)
                    new_c = self.drop(new_c)
                    internal_states[i] = (y,new_c)
            outputs.append(y1)
        #return y,new_c    
        return torch.stack(outputs),(y,new_c)

    def Param_Us(self):
        Us = []
        for c in self.cells:
            Us.append(c.get_U())
        return Us

    def Param_Vs(self):
        Vs = []
        for c in self.cells:
            Vs.append(c.get_V())
        return Vs

    def Param_Sigmas(self):
        Sigmas = []
        for c in self.cells:
            Sigmas.append(c.get_Sigma())
        return Sigmas
    def prun(self,sensitivity):
        for m in self.modules():
            if isinstance(m,SVD_Linear):
                m.prun(sensitivity)




            

        



