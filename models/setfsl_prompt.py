import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
import math
import random
import numpy as np

from utils import split_support_query_set
import backbone.vit_encoder_prompt as vit_encoder


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.o = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        z = F.scaled_dot_product_attention(q, k, v)
        z = self.o(z)
        
        return z

class SETFSL(nn.Module):
    def __init__(self, img_size, patch_size, num_objects, temperature, layer, with_cls=False, continual_layers=None, train_w_qkv=False, train_w_o=False):
        super(SETFSL, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.add_cls_token = True
        self.encoder = self.load_backbone()
         
        self.encoder_dim = self.encoder.embed_dim
        self.ca_dim = self.encoder_dim
        
        self.num_objects = num_objects
        self.temperature = temperature
        self.with_cls = with_cls
        self.continual_layers = continual_layers
        self.train_w_qkv = train_w_qkv
        self.train_w_o = train_w_o
        
        self.layer = layer
        
        self.g_prompt = nn.Parameter(torch.randn(1, self.encoder_dim))
        self.e_prompt = nn.Parameter(torch.randn(1, self.encoder_dim))
        
        self.norm1 = nn.LayerNorm(self.encoder_dim)
        self.norm2 = nn.LayerNorm(self.ca_dim)
        
        self.selfattn = SelfAttention(self.encoder_dim, self.encoder_dim)
        
        for i in range(0, 8):
            for name, param in self.encoder.blocks[i].named_parameters():
                param.requires_grad = False
        
        print('num_object:', num_objects,' temerature:', temperature, ' layer:', layer, ' withcls:', with_cls, ' train_w_qkv:', train_w_qkv, ' train_w_o:', train_w_o,
              'ca_dim:', self.ca_dim)
        
    def load_backbone(self):
        print('img_size: ',self.img_size)
        print('patch_size: ',self.patch_size)
        print('num_patches: ',self.num_patches)
        encoder = vit_encoder.__dict__['vit_small'](img_size=[self.img_size], patch_size=self.patch_size, add_cls_token=True)
        
        return encoder
        
    def forward(self, inputs, labels, device):
        z = self.encoder.forward_gprompt(inputs, self.layer, self.g_prompt)
        
        tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            shots = x_support.size(0) // 5
            x_support_cls = x_support[:, 0, :].unsqueeze(1)
            prototypes_cls = F.normalize(torch.mean(x_support_cls.view(5, shots, self.ca_dim), dim=1), dim=-1) # 5 384
            
            x = torch.concat((prototypes_cls, self.e_prompt), dim=0) # 6 384
            #context_e_prompt = self.selfattn(x)[-1, :].unsqueeze(0)
            context_e_prompt = prototypes_cls
            
            x_support = self.encoder.forward_eprompt(x_support, self.layer, context_e_prompt)[:, 0, :] # cls token
            x_query = self.encoder.forward_eprompt(x_query, self.layer, context_e_prompt)[:, 0, :] # queries 384
            
            prototypes = F.normalize(torch.mean(x_support.view(5, shots, self.ca_dim), dim=1), dim=-1) # supports 5 384
            
            x_query = F.normalize(x_query, dim=-1) # q dim
            distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # q 5
            
            logits = (distance / self.temperature).reshape(-1, 5)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            z = self.encoder.forward_gprompt(inputs, self.layer, self.g_prompt)
            
            tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
            
            loss = 0
            correct = 0
            total = 0
            for x_support, x_query, y_support, y_query in tasks:
                shots = x_support.size(0) // 5
                
                x_support_cls = x_support[:, 0, :].unsqueeze(1)
                prototypes_cls = F.normalize(torch.mean(x_support_cls.view(5, shots, self.ca_dim), dim=1), dim=-1) # 5 384
                
                x = torch.concat((prototypes_cls, self.e_prompt), dim=0) # 6 384
                #context_e_prompt = self.selfattn(x)[-1, :].unsqueeze(0)
                context_e_prompt = prototypes_cls
                
                x_support = self.encoder.forward_eprompt(x_support, self.layer, context_e_prompt)[:, 0, :] # cls token
                x_query = self.encoder.forward_eprompt(x_query, self.layer, context_e_prompt)[:, 0, :] # queries 384
                
                prototypes = F.normalize(torch.mean(x_support.view(5, shots, self.ca_dim), dim=1), dim=-1) # supports 5 384
                
                x_query = F.normalize(x_query, dim=-1) # q dim
                distance = torch.einsum('qd, wd -> qw', x_query, prototypes) # q 5
                
                logits = distance
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc