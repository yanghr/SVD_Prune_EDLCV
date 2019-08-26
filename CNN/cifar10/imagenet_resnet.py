import argparse
import os
import shutil
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from resnet import ResNet
import Regularization
from tqdm import tqdm
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import origin_resnet

parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--reg_type', type=str, default="Hoyer",
                    help='regularization type: 0 Hoyer Hoyer-Square L1')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=0.001, metavar='D',
                    help='weight decay (default: 0.001)')
parser.add_argument('--perp_weight', type=float, default=1.0, metavar='D',
                    help='orthogology restrain weight  (default: 1.0)')
parser.add_argument('--schedule', type=int, nargs='+', default=[31, 61],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')                      
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--save_path', type=str, default='./saves',
                    help='model file')
parser.add_argument('--sensitivity', type=float, default=0.0001,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--depth', type=int, default=50, help='Model depth.')
parser.add_argument('--dectype', type=str, help='the type of decouple', choices=['channel','space'], default='channel')

flag_parser = parser.add_mutually_exclusive_group(required = False)
flag_parser.add_argument("--train",dest = 'train',action = 'store_true')
flag_parser.add_argument("--test",dest = 'train',action = 'store_false')
svd_flag_parser = parser.add_mutually_exclusive_group(required = False)
svd_flag_parser.add_argument("--svd_s1",dest = 'svd_only_stride_1',action = 'store_true')
svd_flag_parser.add_argument("--n_svd_s1",dest = 'svd_only_stride_1',action = 'store_false')
prepruning_parser = parser.add_mutually_exclusive_group(required = False)
prepruning_parser.add_argument("--prun",dest = 'prepruning',action = 'store_true')
prepruning_parser.add_argument("--notprun",dest = 'prepruning',action = 'store_false')
use_origin_pretrain_parser = parser.add_mutually_exclusive_group(required = False)
use_origin_pretrain_parser.add_argument("--origin_pretrain",dest = 'opretrain',action = 'store_true')
use_origin_pretrain_parser.add_argument("--svd_pretrain",dest = 'opretrain',action = 'store_false')
parser.set_defaults(train = True)
parser.set_defaults(svd_only_stride_1 = False)
parser.set_defaults(prepruning = False)
parser.set_defaults(opretrain = False)
parser.add_argument("--load_path",type = str,default = None,help = "where to load a model")
args = parser.parse_args()

# Data
print('==> Preparing dataset ImageNet')
traindir = './data/imagenet/train'
valdir = './data/imagenet/val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_loader = data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


testset = datasets.ImageNet(root = './data',split = 'val',download=True,
                            transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize]))
test_loader = data.DataLoader(testset,batch_size=args.test_batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

# test_loader = data.DataLoader(
#     datasets.ImageFolder(valdir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=args.test_batch_size, shuffle=False,
#     num_workers=args.workers, pin_memory=True)

num_classes = 1000



# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    print("Using %d GPUs"%torch.cuda.device_count())
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Model
print("==> creating model: ResNet-%d"%args.depth)
print("using svd only for stride 1: {}".format(args.svd_only_stride_1))
print("decompose type: "+args.dectype)
print("decay: %f\tsensitivity: %f"%(args.decay,args.sensitivity))
model = torch.nn.DataParallel(ResNet(args.depth,num_classes,args.svd_only_stride_1,args.dectype)).to(device)

if args.load_path is not None:
    if args.opretrain:
        pretrain_model = origin_resnet.ResNet(20,10).to(device)
        pretrain_model.load_state_dict(torch.load(args.load_path)['state_dict'])
        model.init_from_normal_conv(pretrain_model)
    else:
        model.load_state_dict(torch.load(args.load_path))
    # model.standardize_svd()


if args.prepruning:
    print("using prepruning!")
    model.pruning()
# for p in model.parameters():
#     print(p)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = args.lr)

def adjust_learning_rate(optimizer, epoch):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res

def test():
    model.eval()
    test_loss = 0
    top1 = 0
    top5 = 0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            # data = data.view(args.test_batch_size,28*28)
            
            output = model(data)
            test_loss += criterion(output,target).item()
            correct = accuracy(output.data,target.data,topk = (1,5))
            top1 += correct[0]
            top5 += correct[1]
        test_loss /= len(test_loader.dataset)
        top1_accuracy = 100. * top1 / len(test_loader.dataset)
        top5_accuracy = 100. * top5 / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Top1_Accuracy: {top1}/{len(test_loader.dataset)} ({top1_accuracy:.2f}%), Top5_Accuracy: {top5}/{len(test_loader.dataset)} ({top5_accuracy:.2f}%)')
        # perp_loss = 0.0
        # for N in model.GetNs():
        #     perp_loss+=Regularization.orthogology_loss(N,device)
        # for C in model.GetCs():
        #     perp_loss+=Regularization.orthogology_loss(C,device)
        # print("orthogonal loss: {}".format(perp_loss))
    return top1_accuracy,top5_accuracy


def train(epochs):
    log_total_loss = []
    log_reg_loss = []
    log_perp_loss = []
    top1 = []
    top5 = []
    best_acc = 0.0
    model.train()
    for epoch in tqdm(range(epochs),total = epochs):
        adjust_learning_rate(optimizer, epoch)
        for batch_idx,(data,target) in enumerate(train_loader):
            
            data,target = data.to(device),target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            
            perp_loss = 0.0

            for N in model.GetNs():
                perp_loss+=Regularization.orthogology_loss(N,device)
            for C in model.GetCs():
                perp_loss+=Regularization.orthogology_loss(C,device)
            
            reg_loss=Regularization.Reg_Loss(model.GetSigmas(),args.reg_type)
            
            total_loss = loss+reg_loss*args.decay+perp_loss*args.perp_weight
            #print("Model Loss: %f, Reg Loss: %f, Orthogology Loss: %f"%(loss,reg_loss,perp_loss))
            log_total_loss.append(float(total_loss))
            log_reg_loss.append(float(reg_loss))
            log_perp_loss.append(float(perp_loss))
            total_loss.backward()
            optimizer.step()
        top1_accuracy,top5_accuracy = test()
        top1.append(float(top1_accuracy))
        top5.append(float(top5_accuracy))
        torch.save(model.state_dict(),args.save_path+"/SVD_Model.pth")
        if top1_accuracy>best_acc:
            best_acc = top1_accuracy
            shutil.copyfile(args.save_path+"/SVD_Model.pth",args.save_path+"/SVD_BestModel.pth")
    return log_total_loss,log_reg_loss,log_perp_loss,top1,top5





if not os.path.exists(args.save_path+"/images"):
    os.makedirs(args.save_path+"/images",exist_ok=True)
if args.train:
    
    tl_summary,rl_summary,pl_summary,top1_acc,top5_acc = train(args.epochs)
    #model.standardize_svd()
    
    plt.figure()
    plt.subplot(5,1,1)
    plt.plot(tl_summary)
    plt.subplot(5,1,2)
    plt.plot(rl_summary)
    plt.subplot(5,1,3)
    plt.plot(pl_summary)
    plt.subplot(5,1,4)
    plt.plot(top1_acc)
    plt.subplot(5,1,5)
    plt.plot(top5_acc)
    plt.savefig(args.save_path+"/images/learning_curve.png",format = 'png')

    

plt.figure()
singulars = []
for s in model.GetSigmas():
    singulars.append(s.to('cpu').data.numpy())
flat_singulars = np.concatenate(singulars,0)
#print(flat_singulars)
plt.hist(flat_singulars,100)
plt.savefig(args.save_path+"/images/hist.png",format = 'png')


zeros_parms = 0
total_parms = 0
total_energy = 0
prun_energy = 0
for p in model.GetSigmas():
    tensor = p.data.cpu().numpy()
    total_energy+=np.sum(np.square(tensor))
    threshold = args.sensitivity
    new_mask = np.where(abs(tensor) < threshold, 0, tensor)
    zeros_parms += Counter(new_mask.flatten())[0]
    total_parms += new_mask.shape[0]
    prun_energy+=np.sum(np.square(new_mask))
    p.data = torch.from_numpy(new_mask).to(device)


torch.save(model.state_dict(),args.save_path+"/SVD_pruning_Model.pth")    
test()

model.print_modules()
print("zeros parameters: %d, total parameters: %d"%(zeros_parms,total_parms))
print("energy_portion = {}".format(prun_energy/total_energy))
