from __future__ import absolute_import
import torch
import torch.nn as nn
import math
import numpy as np

__all__ = ['resnet']

class SVD_Conv2d(torch.nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,stride = 1,padding = 0,
                bias = False,SVD_only_stride_1 = False,decompose_type = 'channel'):
        """
        stride is fixed to 1 in this module
        """
        super(SVD_Conv2d, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.only_stride_1 = SVD_only_stride_1
        self.decompose_type = decompose_type
        self.output_size = None
        if not SVD_only_stride_1 or self.stride==1:
            if self.decompose_type == 'channel':
                r = min(output_channel,input_channel*kernel_size*kernel_size)
                self.N = torch.nn.Parameter(torch.empty(output_channel,r))#Nxr
                self.C = torch.nn.Parameter(torch.empty(r,input_channel*kernel_size*kernel_size))#rxCHW
                self.Sigma = torch.nn.Parameter(torch.empty(r))#rank = r
            else:#spatial decompose--VH-decompose
                r = min(input_channel*kernel_size,output_channel*kernel_size)
                self.N = torch.nn.Parameter(torch.empty(input_channel*kernel_size,r))#CHxr
                self.C = torch.nn.Parameter(torch.empty(r,output_channel*kernel_size))#rxNW
                self.Sigma = torch.nn.Parameter(torch.empty(r))#rank = r
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
        else:
            self.conv2d = nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding,bias = bias)


    def forward(self,x):
        if not self.only_stride_1 or self.stride==1:
            if self.training:
                r = self.Sigma.size()[0]#r = min(N,CHW)
                # C = self.C[:r, :]#rxCHW
                # N = self.N[:, :r].contiguous()#Nxr
                Sigma = self.Sigma.abs()
                C = torch.mm(torch.diag(torch.sqrt(Sigma)), self.C)
                N = torch.mm(self.N,torch.diag(torch.sqrt(Sigma)))

            
            else:
                valid_idx = torch.arange(self.Sigma.size(0))[self.Sigma!=0]
                N = self.N[:,valid_idx].contiguous()
                C = self.C[valid_idx,:]
                Sigma = self.Sigma[valid_idx].abs()
                r = Sigma.size(0)
                C = torch.mm(torch.diag(torch.sqrt(Sigma)), C)
                N = torch.mm(N,torch.diag(torch.sqrt(Sigma)))
            if self.decompose_type == 'channel':
                C = C.view(r,self.input_channel,self.kernel_size,self.kernel_size)
                N = N.view(self.output_channel,r,1,1)
                y = torch.nn.functional.conv2d(input = x,weight = C,bias = None,stride = self.stride,padding = self.padding)
                y = torch.nn.functional.conv2d(input = y,weight = N,bias = self.bias,stride = 1,padding = 0)
            else:#spatial decompose
                N = N.view(self.input_channel,1,self.kernel_size,r).permute(3,0,2,1)#V:rxcxHx1
                C = C.view(r,self.output_channel,self.kernel_size,1).permute(1,0,3,2)#H:Nxrx1xW
                y = torch.nn.functional.conv2d(input = x,weight = N,bias = None,stride = [self.stride,1],padding = [self.padding,0])
                y = torch.nn.functional.conv2d(input = y,weight = C,bias = self.bias,stride = [1,self.stride],padding = [0,self.padding])

            

        else:
            y = self.conv2d(x)
        self.output_size = y.size()
        #print(self.output_size)
        #input()
        return y
    
    @property
    def ParamN(self):
        if not self.only_stride_1 or self.stride==1:
            return self.N
        else:
            return None
    
    @property
    def ParamC(self):
        if not self.only_stride_1 or self.stride==1:
            return self.C
        else:
            return None

    @property
    def ParamSigma(self):
        if not self.only_stride_1 or self.stride==1:
            return self.Sigma
        else:
            return None


def conv3x3(in_planes, out_planes, stride=1,SVD_only_stride_1 = False,decompose_type = 'channel'):
    "3x3 convolution with padding"
    return SVD_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding = 1,bias=False,SVD_only_stride_1=SVD_only_stride_1,decompose_type = decompose_type)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,downsample=None,SVD_only_stride_1 = False,decompose_type = 'channel'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,SVD_only_stride_1 = SVD_only_stride_1,decompose_type = decompose_type)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,SVD_only_stride_1=SVD_only_stride_1,decompose_type = decompose_type)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
    @property
    def Ns(self):
        N = []
        if self.conv1.ParamN is not None:
            N.append(self.conv1.ParamN)
        if self.conv2.ParamN is not None:
            N.append(self.conv2.ParamN)
        return N

    @property
    def Cs(self):
        C = []
        if self.conv1.ParamC is not None:
            C.append(self.conv1.ParamC)
        if self.conv2.ParamC is not None:
            C.append(self.conv2.ParamC)
        return C

    @property
    def Sigmas(self):
        Sigma = []
        if self.conv1.ParamSigma is not None:
            Sigma.append(self.conv1.ParamSigma)
        if self.conv2.ParamSigma is not None:
            Sigma.append(self.conv2.ParamSigma)
        return Sigma

    



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,SVD_only_stride_1 = False,decompose_type = 'channel'):
        super(Bottleneck, self).__init__()
        self.conv1 = SVD_Conv2d(inplanes, planes, kernel_size=1, bias=False,SVD_only_stride_1 = SVD_only_stride_1,decompose_type = decompose_type)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SVD_Conv2d(planes, planes, kernel_size=3, stride=stride,padding = 1, bias=False,SVD_only_stride_1 = SVD_only_stride_1,decompose_type = decompose_type)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SVD_Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False,SVD_only_stride_1 = SVD_only_stride_1,decompose_type = decompose_type)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    @property
    def Ns(self):
        N = []
        if self.conv1.ParamN is not None:
            N.append(self.conv1.ParamN)
        if self.conv2.ParamN is not None:
            N.append(self.conv2.ParamN)
        if self.conv3.ParamN is not None:
            N.append(self.conv3.ParamN)
        return N

    @property
    def Cs(self):
        C = []
        if self.conv1.ParamC is not None:
            C.append(self.conv1.ParamC)
        if self.conv2.ParamC is not None:
            C.append(self.conv2.ParamC)
        if self.conv3.ParamC is not None:
            C.append(self.conv3.ParamC)
        return C

    @property
    def Sigmas(self):
        Sigma = []
        if self.conv1.ParamSigma is not None:
            Sigma.append(self.conv1.ParamSigma)
        if self.conv2.ParamSigma is not None:
            Sigma.append(self.conv2.ParamSigma)
        if self.conv3.ParamSigma is not None:
            Sigma.append(self.conv3.ParamSigma)
        return Sigma


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000,SVD_only_stride_1 = False,decompose_type = 'channel'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for ImageNet model
        if depth == 18:
            layers = [2,2,2,2]
        elif depth == 34:
            layers = [3,4,6,3]
        elif depth == 50:
            layers = [3,4,6,3]
        elif depth == 101:
            layers = [3,4,23,3]
        elif depth == 152:
            layers = [3,8,36,3]
        else:
            raise ValueError("The depth of ResNet should be one of 18,34,50,101,152")

        block = Bottleneck if depth >=44 else BasicBlock
        self.SVD_only_stride_1 = SVD_only_stride_1
        self.decompose_type = decompose_type
        self.inplanes = 64
        #input : 32x32x3
        self.conv1 = SVD_Conv2d(3, self.inplanes, kernel_size=7, stride= 2,padding = 3,bias=False,SVD_only_stride_1 = SVD_only_stride_1,decompose_type = decompose_type)#32x32x16
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])#32x32x16
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SVD_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False,SVD_only_stride_1 = self.SVD_only_stride_1,decompose_type = self.decompose_type),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,SVD_only_stride_1 = self.SVD_only_stride_1,decompose_type = self.decompose_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,SVD_only_stride_1 = self.SVD_only_stride_1,decompose_type = self.decompose_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32
        x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def GetNs(self):
        Ns = []
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamN is not None:
                Ns.append(m.ParamN)
        return Ns

    def GetCs(self):
        Cs = []
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamC is not None:
                Cs.append(m.ParamC)
        return Cs

    def GetSigmas(self):
        Sigmas = []
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                Sigmas.append(m.ParamSigma)
        return Sigmas

    def print_modules(self):
        i = 0
        FLOPs = 0
        origin_FLOPs = 0
        self.forward(torch.zeros(1,3,224,224).cuda())
        for m in self.modules():
            if isinstance(m,SVD_Conv2d):
                #temp_height = current_size[0]
                current_size = m.output_size
                # current_size[0]=int((current_size[0]+2*m.padding-1*(m.kernel_size-1)-1)/m.stride+1)
                # current_size[1]=int((current_size[1]+2*m.padding-1*(m.kernel_size-1)-1)/m.stride+1)
                feature_size = current_size[2]*current_size[3]
                origin_FLOPs += feature_size*m.kernel_size**2*m.input_channel*m.output_channel
                if m.ParamSigma is not None:
                    rank =  np.count_nonzero(m.ParamSigma.data.cpu().numpy())
                    print("SVD_Conv2d%d:\tinchannel:%d\toutchannel:%d\tkernel_size:%d\tstride:%d\tRank:%d"%(i,m.input_channel,m.output_channel,m.kernel_size,m.stride,rank))
                    if m.decompose_type == 'channel':
                        FLOPs+=feature_size*m.kernel_size**2*m.input_channel*rank
                        FLOPs+=feature_size*1*rank*m.output_channel
                    else:
                        feature_size1 = feature_size*m.stride
                        FLOPs+=feature_size1*m.kernel_size*1*m.input_channel*rank
                        FLOPs+=feature_size*m.kernel_size*1*rank*m.output_channel
                else:
                    print("Normal_Conv2d%d:\tinchannel:%d\toutchannel:%d\tkernel_size:%d\tstride:%d"%(i,m.input_channel,m.output_channel,m.kernel_size,m.stride))
                    FLOPs+=origin_FLOPs
                i+=1
        print("FLOPs:%fM\tOrigin FLOPs:%fM\tSpeedUp: %fx"%(FLOPs/1e6,origin_FLOPs/1e6,origin_FLOPs/FLOPs))

    def pruning(self):
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                valid_idx = torch.arange(m.Sigma.size(0))[m.Sigma!=0]
                m.N.data = m.N[:,valid_idx].contiguous()
                m.C.data = m.C[valid_idx,:]
                m.Sigma.data = m.Sigma[valid_idx]

    def init_from_normal_conv(self,conv_module):
        Ws = []
        Ns = []
        Ss = []
        Cs = []
        Bs = []
        strides = []
        paddings = []
        kernels = []
        for m in conv_module.modules():
            if isinstance(m,torch.nn.Conv2d):
                weight = m.weight.view(m.out_channels,m.in_channels*m.kernel_size[0]*m.kernel_size[1])
                N,S,C = torch.svd(weight, some=True)
                Ws.append(weight)
                Ns.append(N)
                Ss.append(S)
                Cs.append(C)
                Bs.append(m.bias)
                strides.append(m.stride[0])
                paddings.append(m.padding[0])
                kernels.append(m.weight.size(2))
        i = 0
        for m in self.modules():
            if isinstance(m,SVD_Conv2d):
                if m.ParamSigma is not None:
                    m.N.data = Ns[i]
                    m.Sigma.data = Ss[i]
                    m.C.data = Cs[i].transpose(0,1)
                    m.bias = Bs[i]
                    print(torch.sum((m.N.mm(m.Sigma.diag()).mm(m.C)-Ws[i])**2))
                else:
                    m.conv2d.weight = Ws[i]
                    m.conv2d.bias = Bs[i]
                m.stride = strides[i]
                m.padding = paddings[i]
                m.kernel_size = kernels[i]
                i+=1
            

    def standardize_svd(self):
        for m in self.modules():
            if isinstance(m,SVD_Conv2d) and m.ParamSigma is not None:
                NC = m.N.matmul(m.Sigma.abs().diag()).matmul(m.C)
                N,S,C = torch.svd(NC,some = True)
                C = C.transpose(0,1)
                m.N.data = N
                m.C.data = C
                m.Sigma.data = S

                



