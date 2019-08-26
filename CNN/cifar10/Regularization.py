import torch

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