import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
import math
import numpy as np

from utils import split_support_query_set
import backbone.vit_encoder as vit_encoder

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, y):
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        
        z = F.scaled_dot_product_attention(q, k, v)
        z = self.o(z)
        
        return z

class SET_FSL(nn.Module):
    def __init__(self, img_size, patch_size, num_objects, temperature, layer, with_cls=False, continual_layers=None, train_w_qkv=False, train_w_o=False):
        super(SET_FSL, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        
        self.add_cls_token = True
        self.encoder = self.load_backbone()
        self.outdim = self.encoder.embed_dim
        self.num_objects = num_objects
        self.temperature = temperature
        self.with_cls = with_cls
        self.continual_layers = continual_layers
        self.train_w_qkv = train_w_qkv
        self.train_w_o = train_w_o
        
        self.object_queries = nn.Embedding(self.num_objects, self.encoder.embed_dim)
        self.layer = layer
        
        self.norm1 = nn.LayerNorm(self.outdim)
        self.norm2 = nn.LayerNorm(self.outdim)
        self.crossattn = CrossAttention(self.outdim)
        
        if continual_layers != None:
            self.ca_blocks = nn.ModuleList([CrossAttention(self.outdim) for i in range(len(continual_layers))])
            self.layer_nums = continual_layers
            print(*self.layer_nums)
        
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        
        print('num_object:', num_objects,' temerature:', temperature, ' layer:', layer, ' withcls:', with_cls, ' train_w_qkv:', train_w_qkv, ' train_w_o:', train_w_o)
        
    def load_backbone(self):
        print('img_size: ',self.img_size)
        print('patch_size: ',self.patch_size)
        print('num_patches: ',self.num_patches)
        encoder = vit_encoder.__dict__['vit_small'](img_size=[self.img_size], patch_size=self.patch_size, add_cls_token=True)
        
        return encoder
    
    def make_qkv_identity(self, crossattn):
        crossattn.q = nn.Identity()
        crossattn.k = nn.Identity()
        crossattn.v = nn.Identity()
    
    def make_o_identity(self, crossattn):
        crossattn.o = nn.Identity()
    
    def individual_crossattn(self, z):
        if not self.train_w_qkv:
            self.make_qkv_identity(self.crossattn)
        if not self.train_w_o:
            self.make_o_identity(self.crossattn)
            
        q = self.object_queries.weight
        kv = z[self.layer]
        z = self.crossattn(q, kv)
        
        return z
    
    def continual_crossattn(self, z):
        object_queries = torch.zeros_like(self.object_queries.weight)
        object_queries = object_queries + self.object_queries.weight
        
        self.make_o_identity(self.crossattn)
        for blk, l in zip(self.ca_blocks, self.layer_nums):
            if not self.train_w_qkv:
                self.make_qkv_identity(blk)
            if not self.train_w_o:
                self.make_o_identity(blk)
            q = object_queries
            kv = z[l]
            x = self.crossattn(q, kv)
            
            object_queries = object_queries + x
            
        return x
        
    def forward(self, inputs, labels, device):
        with torch.no_grad():
            z = self.norm1(self.encoder(inputs, return_attn=False, memories=True))
        
        cls_token = z[self.layer][:, 0, :].unsqueeze(1)
        
        if self.continual_layers == None:
            # test 1
            z = self.individual_crossattn(z)
        else:
            # test 2
            z = self.continual_crossattn(z)
        z = self.norm2(z)
        
        num_objects = self.num_objects
        if self.with_cls:
            z = torch.concat((z, cls_token), dim=1)
            num_objects = self.num_objects + 1
        
        tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            x_query = F.normalize(x_query, dim=-1)
            shots = x_support.size(0) // 5
            prototypes = F.normalize(torch.mean(x_support.view(5, shots, -1, self.outdim), dim=1), dim=-1)
            
            x_query = x_query.transpose(0, 1) # objects queries dim
            prototypes = prototypes.transpose(0, 1) # objects 5 dim
            
            distance = torch.einsum('oqd, owd -> oqw', x_query, prototypes).transpose(0, 1) # queries objects 5
            
            y_query = y_query.unsqueeze(1).repeat(1, num_objects).view(-1)
            logits = (distance / self.temperature).reshape(-1, 5)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            
            z = self.norm1(self.encoder(inputs, return_attn=False, memories=True))
            
            cls_token = z[self.layer][:, 0, :].unsqueeze(1)
            
            if self.continual_layers == None:
                z = self.individual_crossattn(z)
            else:
                z = self.continual_crossattn(z)
            z = self.norm2(z)
            
            if self.with_cls:
                z = torch.concat((z, cls_token), dim=1)
                
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
                
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc
