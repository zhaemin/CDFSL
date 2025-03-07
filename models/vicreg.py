import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
from tqdm import tqdm
import numpy as np

import backbone.encoders as encoders
from models.SSL import SSLFramework

from utils import split_support_query_set, mixup

class VICReg(SSLFramework):
    def __init__(self, backbone, mixup):
        super(VICReg, self).__init__(backbone)
        self.projector =  self.make_mlp(self.outdim, hidden_dim=2048, num_layers=3, out_dim=2048, last_bn=False)
        self.num_features = 2048
        self.mixup = mixup

    def forward(self, inputs, device):
        x = inputs[0]
        y = inputs[1]
        batch_size = x.size(0)
        
        x = self.projector(self.encoder(x))
        y = self.projector(self.encoder(y))
        
        if self.mixup:
            z_mixup, mixup_ind, lam = mixup(inputs[2], alpha=1.)
            z_mixup = self.projector(self.encoder(z_mixup))
            
            feature_mixup = x + y / 2
            lam_expanded = lam.view([-1] + [1] * (feature_mixup.dim()-1)) # lam -> b, 1
            mixup_feature = lam_expanded * feature_mixup + (1. - lam_expanded) * feature_mixup[mixup_ind]
            # mixup loss
            mixup_loss = F.mse_loss(z_mixup, mixup_feature)
        
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        
        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        
        loss = (25 * repr_loss + 25 * std_loss + 1 * cov_loss)
        if self.mixup:
            loss += mixup_loss
        
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
