import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
#import torchsnooper
from collections import Counter

class FCModel(torch.nn.Module):
    def __init__(self,input_dim = 28*28,output_dim = 10,
        Hidden_Layer = [300,100],activation_fn = torch.nn.ReLU(),
        output_fn = torch.nn.LogSoftmax(dim = 1),device = 'cpu'):
        super(FCModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.output_fn = output_fn
        self.input_layer = torch.nn.Linear(self.input_dim,Hidden_Layer[0])
        self.hidden_net = [torch.nn.Linear(Hidden_Layer[i],Hidden_Layer[i+1]).to(device) for i in range(len(Hidden_Layer)-1)]
        self.output_layer = torch.nn.Linear(Hidden_Layer[-1],self.output_dim)
    def forward(self,x):
        y = self.input_layer(x)
        y = self.activation_fn(y)
        for L in self.hidden_net:
            y = L(y)
            y = self.activation_fn(y)
        y = self.output_layer(y)
        y = self.output_fn(y)
        return y

def Reg_Loss(parameters,reg_type = 'Hoyer'):
    """
    type can be : Hoyer,Hoyer-Square,L1
    """
    reg = 0.0
    for param in parameters:
        if param.requires_grad and torch.sum(torch.abs(param))>0:
            if reg_type == "Hoyer":
                reg += torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2))-1#Hoyer
            elif reg_type == "Hoyer-Square":
                reg += (torch.sum(torch.abs(param))**2)/torch.sum(param**2)-1#Hoyer-Square
            elif reg_type == "L1":    
                reg += torch.sum(torch.abs(param))#L1
            else:
                reg = 0.0
    return reg

def Reg_Loss_Param(param,reg_type = 'Hoyer'):
    """
    Regularization for single parameter
    """
    reg = 0.0
    if param.requires_grad and torch.sum(torch.abs(param))>0:
        if reg_type == "Hoyer":
            reg = torch.sum(torch.abs(param))/torch.sqrt(torch.sum(param**2))-1#Hoyer
        elif reg_type == "Hoyer-Square":
            reg = (torch.sum(torch.abs(param))**2)/torch.sum(param**2)-1#Hoyer-Square
        elif reg_type == "L1":    
            reg = torch.sum(torch.abs(param))#L1
        else:
            reg = 0.0
    return reg

def orthogology_loss(mat,device = 'cpu'):
    loss = 0.0
    if mat.requires_grad:
        if mat.size(0)<=mat.size(1):
            mulmat = mat.matmul(mat.transpose(0,1))#AxA'
        else:
            mulmat = mat.transpose(0,1).matmul(mat)#A'xA
        loss = torch.sum((mulmat-torch.eye(mulmat.size(0),device = device))**2)/(mulmat.size(0)*mulmat.size(1))
    return loss


class SVD_FCModel(torch.nn.Module):
    def __init__(self,input_dim = 28*28,output_dim = 10,
        Hidden_Layer = [300,100],activation_fn = torch.nn.ReLU(),
        output_fn = torch.nn.LogSoftmax(dim = 1),device = 'cpu'):
        super(SVD_FCModel, self).__init__()
        self.device = device
        self.activation_fn = activation_fn
        self.output_fn = output_fn
        Hidden_Layer.append(output_dim)
        Hidden_Layer.insert(0,input_dim)
        # input_U = torch.empty(input_dim,input_dim,device = device)
        # input_Singular = torch.empty(min(input_dim,Hidden_Layer[0]),device = device)
        # input_V = torch.empty(Hidden_Layer[0],Hidden_Layer[0],device = device)
        # input_b = torch.empty(Hidden_Layer[0],device = device)
        self.Us = []
        self.Singulars = []
        self.Vs = []
        self.bs = []
        for i in range(len(Hidden_Layer)-1):
            r = min(Hidden_Layer[i],Hidden_Layer[i+1])
            self.Us.append(torch.empty(Hidden_Layer[i],r,device = device))
            self.Singulars.append(torch.empty(r,device = device))
            self.Vs.append(torch.empty(r,Hidden_Layer[i+1],device = device))
            self.bs.append(torch.empty(Hidden_Layer[i+1],device = device))
        for i in range(len(self.Us)):
            #register and init the parameters
            self.Us[i] = torch.nn.Parameter(self.Us[i])
            self.Vs[i] = torch.nn.Parameter(self.Vs[i])
            self.Singulars[i] = torch.nn.Parameter(self.Singulars[i])
            self.bs[i] = torch.nn.Parameter(self.bs[i])
            self.register_parameter('U_%d'%i,self.Us[i])
            self.register_parameter('V_%d'%i,self.Vs[i])
            self.register_parameter('Singular_Value_%d'%i,self.Singulars[i])
            self.register_parameter('Bias_%d'%i,self.bs[i])
            torch.nn.init.xavier_normal_(self.Us[i])
            torch.nn.init.xavier_normal_(self.Vs[i])
            torch.nn.init.uniform_(self.Singulars[i])
            torch.nn.init.constant_(self.bs[i],0.0)

    @property
    def layer_depth(self):
        return len(self.Us)
    
    def forward(self,x):
        y = x
        for i in range(len(self.Us)):
            if self.training:
                y = y.matmul(self.Us[i])
                y = y.matmul(self.Singulars[i].abs().diag())
                y = y.matmul(self.Vs[i])
            else:
                valid_idx = torch.arange(self.Singulars[i].size(0))[self.Singulars[i]!=0]
                U = self.Us[i][:,valid_idx].contiguous()
                V = self.Vs[i][valid_idx,:]
                Sigma = self.Singulars[i][valid_idx]
                y = y.matmul(U)
                y = y.matmul(Sigma.abs().diag())
                y = y.matmul(V)
            y = y.add(self.bs[i])
            if i != len(self.Us)-1:
                y = self.activation_fn(y)
            else:
                y = self.output_fn(y)
        return y
    


class SVD_Conv2d(torch.nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,
                bias = True,sensitivity = 1e-3):
        """
        stride is fixed to 1 in this module
        """
        super(SVD_Conv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.sensitivity = sensitivity


        self.N = torch.nn.Parameter(torch.empty(output_channel,output_channel))#NxN
        self.C = torch.nn.Parameter(torch.empty(input_channel*kernel_size*kernel_size,input_channel*kernel_size*kernel_size))#CHWxCHW
        self.Sigma = torch.nn.Parameter(torch.empty(min(output_channel,input_channel*kernel_size*kernel_size)))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(output_channel))
            self.register_parameter('bias',self.bias)
            torch.nn.init.constant_(self.bias,0.0)
        self.register_parameter('N',self.N)
        self.register_parameter('C',self.C)
        self.register_parameter('Sigma',self.Sigma)
        torch.nn.init.kaiming_normal_(self.N)
        torch.nn.init.kaiming_normal_(self.C)
        torch.nn.init.normal_(self.Sigma)


    def forward(self,x):
        if self.training:
            r = self.Sigma.size()[0]#r = min(N,CHW)
            C = self.C[:r, :]#rxCHW
            N = self.N[:, :r].contiguous()#Nxr
            C = torch.mm(torch.diag(torch.sqrt(self.Sigma)), C)
            N = torch.mm(N,torch.diag(torch.sqrt(self.Sigma)))
            
        
        else:
            valid_idx = torch.arange(self.Sigma.size(0))[torch.abs(self.Sigma)>self.sensitivity]
            N = self.N[:,valid_idx].contiguous()
            C = self.C[valid_idx,:]
            Sigma = self.Sigma[valid_idx]
            r = Sigma.size(0)
            C = torch.mm(torch.diag(torch.sqrt(Sigma)), C)
            N = torch.mm(N,torch.diag(torch.sqrt(Sigma)))

        C = C.view(r,self.input_channel,self.kernel_size,self.kernel_size)
        N = N.view(self.output_channel,r,1,1)
        y = torch.nn.functional.conv2d(input = x,weight = C,bias = None,stride = 1,padding = 1)
        y = torch.nn.functional.conv2d(input = y,weight = N,bias = self.bias,stride = 1,padding = 0)
        return y
        


    

parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--reg_type', type=str, default="Hoyer",
                    help='regularization type: 0 Hoyer Hoyer-Square L1')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.001, metavar='D',
                    help='weight decay (default: 0.001)')
parser.add_argument('--perp_weight', type=float, default=0.1, metavar='D',
                    help='orthogology restrain weight  (default: 0.1)')                      
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12345678, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--save_path', type=str, default='./saves',
                    help='model file')
parser.add_argument('--sensitivity', type=float, default=0.0001,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
#parser.add_argument("--training",type = bool,default = True,help = "whether to train the model")
flag_parser = parser.add_mutually_exclusive_group(required = False)
flag_parser.add_argument("--train",dest = 'train',action = 'store_true')
flag_parser.add_argument("--test",dest = 'train',action = 'store_false')
parser.set_defaults(train = True)
parser.add_argument("--load_path",type = str,default = None,help = "where to load a model")
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs
test_batch_size = args.test_batch_size
reg_type = args.reg_type
reg_weight = args.decay
perp_weight = args.perp_weight
lr = args.lr
hidden_layer = [300,100]#LeNet-300-100
epsilon = args.sensitivity

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(123)
else:
    print('Not using CUDA!!!')


# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=test_batch_size, shuffle=False, **kwargs)

model = SVD_FCModel(input_dim = 28*28,output_dim = 10,Hidden_Layer = hidden_layer,device = device).to(device)

if args.load_path is not None:
    model.load_state_dict(torch.load(args.load_path))

# for p in model.parameters():
#     print(p)
optimizer = optim.Adam(model.parameters(),lr = lr)

#@torchsnooper.snoop()
def train(epochs):
    log_total_loss = []
    log_reg_loss = []
    model.train()
    for epoch in tqdm(range(epochs),total = epochs):#tqdm(range(epochs),total = epochs):
        for batch_idx,(data,target) in enumerate(train_loader):
            
            data,target = data.to(device),target.to(device)
            data = data.view(batch_size,28*28)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output,target)
            reg = Reg_Loss(model.parameters(),reg_type)
            total_loss = loss+reg*reg_weight
            # print("Model Loss: %f, Reg Loss: %f"%(loss,reg))
            log_total_loss.append(total_loss)
            log_reg_loss.append(reg)
            total_loss.backward()
            optimizer.step()
    return log_total_loss,log_reg_loss

def train_SVD(epochs):
    log_total_loss = []
    log_reg_loss = []
    log_perp_loss = []
    model.train()
    for epoch in tqdm(range(epochs),total = epochs):#tqdm(range(epochs),total = epochs):
        for batch_idx,(data,target) in enumerate(train_loader):
            
            data,target = data.to(device),target.to(device)
            data = data.view(batch_size,28*28)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output,target)
            perp_loss = 0.0
            reg_loss = 0.0
            for i in range(model.layer_depth):
                perp_loss += (orthogology_loss(model.Us[i],device)+orthogology_loss(model.Vs[i],device))
                reg_loss += (Reg_Loss_Param(model.Singulars[i],reg_type)+Reg_Loss_Param(model.bs[i],reg_type))
            total_loss = loss+reg_loss*reg_weight+perp_loss*perp_weight
            #print("Model Loss: %f, Reg Loss: %f, Orthogology Loss: %f"%(loss,reg_loss,perp_loss))
            log_total_loss.append(total_loss)
            log_reg_loss.append(reg_loss)
            log_perp_loss.append(perp_loss)
            total_loss.backward()
            optimizer.step()
    return log_total_loss,log_reg_loss,log_perp_loss

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(test_batch_size,28*28)
            
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output,target,reduction='sum').item()
            pred = output.data.max(1,keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

if not os.path.exists(args.save_path+"/images"):
    os.makedirs(args.save_path+"/images",exist_ok=True)
if args.train:
    
    tl_summary,rl_summary,pl_summary = train_SVD(epochs)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(tl_summary)
    plt.subplot(3,1,2)
    plt.plot(rl_summary)
    plt.subplot(3,1,3)
    plt.plot(pl_summary)
    plt.savefig(args.save_path+"/images/learning_curve.png",format = 'png')

    torch.save(model.state_dict(),args.save_path+"/SVD_Model.pth")

plt.figure()
singulars = []
for i in range(len(model.Singulars)):
    singulars.append(model.Singulars[i].to('cpu').data.numpy())
flat_singulars = np.concatenate(singulars,0)

plt.hist(flat_singulars,100)
plt.savefig(args.save_path+"/images/hist.png",format = 'png')

accuracy = test()
zeros_parms = 0
total_parms = 0
for p in model.Singulars:
    tensor = p.data.cpu().numpy()
    threshold = epsilon
    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
    zeros_parms += Counter(new_mask.flatten())[0]
    total_parms += new_mask.shape[0]
    p.data = torch.from_numpy(new_mask).to(device)
    
accuracy = test()
print("zeros parameters: %d, total parameters: %d"%(zeros_parms,total_parms))
