import torch
import torch.nn as nn
from torch.autograd import Variable


class SVD_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers = 1,bias = True,batch_first = False,dropout = 0,bidirectional = False,*args):
        super(SVD_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_size,
                            num_layers = num_layers,bias = bias,batch_first = batch_first,dropout = dropout,
                            bidirectional = bidirectional)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.weight_ih = []
        self.weight_hh = []
        
        for name,p in self.lstm.named_parameters():
            if 'weight_ih' in name:
                self.weight_ih.append(p)
            elif 'weight_hh' in name:
                self.weight_hh.append(p)

        self.weight_ih_U = nn.ParameterList()
        self.weight_ih_Singular = nn.ParameterList()
        self.weight_ih_V = nn.ParameterList()
        self.weight_hh_U = nn.ParameterList()
        self.weight_hh_Singular = nn.ParameterList()
        self.weight_hh_V = nn.ParameterList()
        self.mm_weight_ih = []
        self.mm_weight_hh = []
        for l in range(len(self.weight_ih)):
            u,s,v = self.weight_ih[l].svd()
            self.weight_ih_U.append(nn.Parameter(u.clone().detach()))
            self.weight_ih_V.append(nn.Parameter(v.transpose(0,1).clone().detach()))
            self.weight_ih_Singular.append(nn.Parameter(s.clone().detach()))
            u1,s1,v1 = self.weight_hh[l].svd()
            self.weight_hh_U.append(nn.Parameter(u1.clone().detach()))
            self.weight_hh_V.append(nn.Parameter(v1.transpose(0,1).clone().detach()))
            self.weight_hh_Singular.append(nn.Parameter(s1.clone().detach()))
            self.mm_weight_ih.append(self.weight_ih[l].data)
            self.mm_weight_hh.append(self.weight_hh[l].data)
        #self.register_backward_hook(self.svd_hook)
        
    def forward(self,x,hidden):
        for l in range(len(self.weight_ih)):
            self.mm_weight_ih[l] = self.weight_ih_U[l].mm(self.weight_ih_Singular[l].abs().diag()).mm(self.weight_ih_V[l])
            self.mm_weight_hh[l] = self.weight_hh_U[l].mm(self.weight_hh_Singular[l].abs().diag()).mm(self.weight_hh_V[l])
            self.weight_ih[l].data = self.mm_weight_ih[l]
            self.weight_hh[l].data = self.mm_weight_hh[l]
        self.lstm.flatten_parameters()
        return self.lstm(x,hidden)
    
    def svd_grad(self):
        for l in range(len(self.weight_ih)):
            self.mm_weight_ih[l].backward(self.weight_ih[l].grad.data)
            self.mm_weight_hh[l].backward(self.weight_hh[l].grad.data)

    # def svd_hook(self,module,grad_input,grad_output):
    #     self.svd_grad()
    #     return grad_input
    def prun(self,sensitivity):
        assert sensitivity <= 1.0 and sensitivity >= 0.0
        for l in range(len(self.weight_ih)):
            #ih
            energy = self.weight_ih_Singular[l]**2
            _,sorted_idx = torch.sort(energy,descending = True)
            sum_e = torch.sum(energy)
            valid = 0
            for i in range(len(sorted_idx)):
                if energy[sorted_idx[:(i+1)]].sum()/sum_e >= sensitivity:
                    valid = i+1
                    break
            valid_idx = sorted_idx[:valid]
            self.weight_ih_U[l].data = self.weight_ih_U[l][:,valid_idx]
            self.weight_ih_V[l].data = self.weight_ih_V[l][valid_idx,:]
            self.weight_ih_Singular[l].data = self.weight_ih_Singular[l][valid_idx]

            #hh
            energy = self.weight_hh_Singular[l]**2
            _,sorted_idx = torch.sort(energy,descending = True)
            sum_e = torch.sum(energy)
            valid = 0
            for i in range(len(sorted_idx)):
                if energy[sorted_idx[:(i+1)]].sum()/sum_e >= sensitivity:
                    valid = i+1
                    break
            valid_idx = sorted_idx[:valid]
            self.weight_hh_U[l].data = self.weight_hh_U[l][:,valid_idx]
            self.weight_hh_V[l].data = self.weight_hh_V[l][valid_idx,:]
            self.weight_hh_Singular[l].data = self.weight_hh_Singular[l][valid_idx]
            self.mm_weight_ih[l] = self.weight_ih_U[l].mm(self.weight_ih_Singular[l].abs().diag()).mm(self.weight_ih_V[l])
            self.mm_weight_hh[l] = self.weight_hh_U[l].mm(self.weight_hh_Singular[l].abs().diag()).mm(self.weight_hh_V[l])
            self.weight_ih[l].data = self.mm_weight_ih[l]
            self.weight_hh[l].data = self.mm_weight_hh[l]

        
# lstm = SVD_LSTM(3,8,2)
# print(lstm.weight_hh)
# data = torch.zeros(5,2,3)
# lstm(data,[Variable(torch.zeros(2,2,8)),
#                 Variable(torch.zeros(2,2,8))])
# print(lstm.weight_hh)
