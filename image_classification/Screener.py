import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict


class Screener_MNIST(nn.Module):
    def __init__(self):
        super(Screener_MNIST, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(1,4,3)),('act1',nn.ELU()), # 28-3+1=26
            ('conv2',nn.Conv2d(4,8,3)),('act2',nn.ELU()), # 26-3+1=24
            ('conv3',nn.Conv2d(8,16,3)),('act3',nn.ELU()), # 24-3+1=22
            ('conv4',nn.Conv2d(16,32,3)),('act4',nn.ELU())])) # 22-3+1=20
        self.fc = nn.Linear(20*20*32,1)
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 20*20*32)
        out = F.sigmoid(self.fc(x))
        return out

    def snet_loss(self, snet_out, cls_err, variables, M=1.0, alpha=0.01):
        w = None
        for p in variables:
            w = torch.cat((w, p.view(-1))) if w is not None else p.view(-1)
        l1 = F.l1_loss(w, torch.zeros_like(w))
        loss = torch.pow((1-snet_out),2)
        loss= loss*cls_err
        loss = loss + torch.pow(snet_out,2)*torch.clamp(M-cls_err, min=0)
        res = torch.sum(loss)+alpha*l1
        return res


class Screener_CIFAR10(nn.Module):
    def __init__(self):
        super(Screener_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.act(self.fc3(x))
        return x
    
    def snet_loss(self, snet_out, cls_err, variables, M=10, alpha=0.01):
        w = None
        for p in variables:
            w = torch.cat((w, p.view(-1))) if w is not None else p.view(-1)
        l1 = F.l1_loss(w, torch.zeros_like(w))
        loss = torch.pow((1-snet_out),2)
        loss= loss*cls_err
        loss = loss + torch.pow(snet_out,2)*torch.clamp(M-cls_err, min=0)
        res = torch.sum(loss)+alpha*l1
        return res

class Screener_ImageNet(nn.Module):
    def __init__(self):
        super(Screener_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.act(self.fc3(x))
        return x
    
    def snet_loss(self, snet_out, cls_err, variables, M=10, alpha=0.01):
        w = None
        for p in variables:
            w = torch.cat((w, p.view(-1))) if w is not None else p.view(-1)
        l1 = F.l1_loss(w, torch.zeros_like(w))
        loss = torch.pow((1-snet_out),2)
        loss= loss*cls_err
        loss = loss + torch.pow(snet_out,2)*torch.clamp(M-cls_err, min=0)
        res = torch.sum(loss)+alpha*l1
        return res
