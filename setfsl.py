import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
import math
import numpy as np

from utils import split_support_query_set
from backbone import visiontransformer as vit

from tqdm import tqdm


class SET_FSL(nn.Module):
    def __init__(self, img_size, patch_size, num_objects, temperature, attn, layer):
        super(SET_FSL, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.add_cls_token = True
        self.encoder, self.decoder = self.load_backbone()
        self.outdim = self.encoder.embed_dim
        self.num_objects = num_objects
        self.temperature = temperature
        
        self.query_pos = nn.Embedding(self.num_objects, self.encoder.embed_dim)
        self.attn = attn
        self.layer = layer
        
        for name, param in self.encoder.named_parameters():
            if name != 'weight_for_blk':
                param.requires_grad = False
        
        print('num_object:', num_objects,' temerature:', temperature, ' attn:', attn, ' layer:', layer)
        
    def load_backbone(self):
        print('img_size: ',self.img_size)
        print('patch_size: ',self.patch_size)
        print('num_patches: ',self.num_patches)
        encoder = vit.__dict__['vit_small'](img_size=[self.img_size], patch_size=self.patch_size, add_cls_token=True)
        decoder = vit.__dict__['vit_decoder'](self.num_patches, encoder.embed_dim, encoder.num_heads)
        
        return encoder, decoder
        
    def forward(self, inputs, labels, device):
        with torch.no_grad():
            z = self.encoder(inputs, return_attn=True, find_optimal_target=True)
        z = self.decoder(self.query_pos.weight, z, self.attn, self.layer) # bs 1 dim
        
        tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            x_query = F.normalize(x_query, dim=-1)
            shots = x_support.size(0) // 5
            prototypes = F.normalize(torch.mean(x_support.view(5, shots, -1, self.outdim), dim=1), dim=-1)
            
            x_query = x_query.transpose(0, 1) # objects queries dim
            prototypes = prototypes.transpose(0, 1) # objects 5 dim
            
            distance = torch.einsum('oqd, owd -> oqw', x_query, prototypes).transpose(0, 1) # queries objects 5
            #logits = distance.min(dim=1).values / self.temperature
            
            y_query = y_query.unsqueeze(1).repeat(1, self.num_objects+1).view(-1)
            logits = (distance / self.temperature).reshape(-1, 5)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            
            z = self.encoder(inputs, return_attn=True, find_optimal_target=True)
            z = self.decoder(self.query_pos.weight, z, self.attn, self.layer) # bs 1 dim

            tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
            
            loss = 0
            for x_support, x_query, y_support, y_query in tasks:
                x_query = F.normalize(x_query, dim=-1)
                shots = x_support.size(0) // 5
                prototypes = F.normalize(torch.mean(x_support.view(5, shots, -1, self.outdim), dim=1), dim=-1)
                
                x_query = x_query.transpose(0, 1) # objects queries dim
                prototypes = prototypes.transpose(0, 1) # objects 5 dim
                
                distance = torch.einsum('oqd, owd -> oqw', x_query, prototypes).transpose(0, 1) # queries objects 5
                
                new_distance = []
                for i in range(distance.size(0)):
                    mean = distance[i].mean(dim=1, keepdim=True)
                    var = distance[i].var(dim=1, keepdim=True)
                    nd = (distance[i] - mean) / (var + 1.e-6)**.5
                    new_distance.append(nd)
            
                distance = torch.stack(new_distance)
                logits = distance.mean(dim=1)
                
                #logits = distance.min(dim=1).values
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc