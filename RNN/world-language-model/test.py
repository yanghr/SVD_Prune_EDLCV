import torch

lstm = torch.nn.LSTM(1,3,3)
weight_ih = []
weight_hh = []
bias_ih = []
bias_hh = []
for i,p in enumerate(lstm.parameters()):
    if i%4==0:
        weight_ih.append(p)
    elif i%4==1:
        weight_hh.append(p)
    elif i%4==2:
        bias_ih.append(p)
    else:
        bias_hh.append(p)
weight_ih[1][-1,-1] = 10.0
for p in lstm.parameters():
    print(p)