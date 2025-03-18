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
import backbone.vit_encoder as vit_encoder

class CrossAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.o = nn.Linear(output_dim, output_dim)
    
    def forward(self, x, y):
        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        
        z = F.scaled_dot_product_attention(q, k, v)
        z = self.o(z)
        
        return z

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
        
        self.object_queries = nn.Embedding(self.num_objects, self.encoder_dim)
        self.layer = layer
        
        self.norm1 = nn.LayerNorm(self.encoder_dim)
        self.norm2 = nn.LayerNorm(self.ca_dim)
        
        self.selfattn = SelfAttention(self.encoder_dim, self.encoder_dim)
        
        # continual CA
        if continual_layers != None:
            self.object_queries = nn.Embedding(self.num_objects, self.encoder_dim)
            self.ca_blocks = nn.ModuleList([CrossAttention(self.encoder_dim, self.ca_dim) for i in range(len(continual_layers))])
            self.layer_nums = continual_layers
            print('continual layers:', *self.layer_nums)
            
            for blk in self.ca_blocks:
                if not self.train_w_qkv:
                    self.make_qkv_identity(blk)
                if not self.train_w_o:
                    self.make_o_identity(blk)
        # individual CA
        else:
            self.crossattn = CrossAttention(self.encoder_dim, self.ca_dim)
            if not self.train_w_qkv:
                self.make_qkv_identity(self.crossattn)
            if not self.train_w_o:
                self.make_o_identity(self.crossattn)
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        print('num_object:', num_objects,' temerature:', temperature, ' layer:', layer, ' withcls:', with_cls, ' train_w_qkv:', train_w_qkv, ' train_w_o:', train_w_o,
              'ca_dim:', self.ca_dim)
        
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
    
    def layernorm_affine_false(self):
        self.encoder.norm
    
    def individual_crossattn(self, z, object_queries):
        q = object_queries
        kv = z[self.layer]
        x = self.crossattn(q, kv)
        
        return x
    
    def continual_crossattn(self, z, object_queries):
        x = object_queries
        
        for blk, l in zip(self.ca_blocks, self.layer_nums):
            q = x
            kv = z[l]
            x = blk(q, kv)
            
            x = x + q
            
        return x
        
    def forward(self, inputs, labels, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=5)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            x_support = self.encoder(x_support, return_attn=False, memories=True)
            x_query = self.encoder(x_query, return_attn=False, memories=True)
            
            shots = x_support.size(1) // 5
            x_support_cls = x_support[self.layer][:, 0, :].unsqueeze(1)
            prototypes_cls = F.normalize(torch.mean(x_support_cls.view(5, shots, self.ca_dim), dim=1), dim=-1) # 5 384
            
            x = torch.concat((prototypes_cls, self.object_queries.weight), dim=0) # 6 384
            #context_object_queries = self.selfattn(x)[-1, :].unsqueeze(0)
            context_object_queries = x
            
            if self.continual_layers == None:
                # test 1
                x_query = self.individual_crossattn(x_query, context_object_queries)
                x_support = self.individual_crossattn(x_support, context_object_queries) # supports 1 384
            else:
                # test 2
                x_query = self.continual_crossattn(x_query, context_object_queries)
                x_support = self.continual_crossattn(x_support, context_object_queries)
            x_query = self.norm2(x_query) # 75 1 384
            x_support = self.norm2(x_support)
            #x_query = self.norm2(x_query[:, -1, :]) # 75 1 384
            #x_support = self.norm2(x_support[:, -1, :])
            
            num_objects = self.num_objects
            
            prototypes = F.normalize(torch.mean(x_support.view(5, shots, self.ca_dim), dim=1), dim=-1) # 5 384
            x_query = x_query.transpose(0, 1) # objects queries dim
            prototypes = prototypes.unsqueeze(1).transpose(0, 1) # objects 5 dim
            
            x_query = F.normalize(x_query, dim=-1)
            distance = torch.einsum('oqd, owd -> oqw', x_query, prototypes).transpose(0, 1) # queries objects 5
            
            y_query = y_query.unsqueeze(1).repeat(1, num_objects).view(-1)
            logits = (distance / self.temperature).reshape(-1, 5)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=5)
            
            loss = 0
            correct = 0
            total = 0
            for x_support, x_query, y_support, y_query in tasks:
                x_support = self.encoder(x_support, return_attn=False, memories=True)
                x_query = self.encoder(x_query, return_attn=False, memories=True)
                
                shots = x_support.size(1) // 5
                x_support_cls = x_support[self.layer][:, 0, :].unsqueeze(1)
                prototypes_cls = F.normalize(torch.mean(x_support_cls.view(5, shots, self.ca_dim), dim=1), dim=-1) # 5 384
                
                x = torch.concat((prototypes_cls, self.object_queries.weight), dim=0) # 6 384
                #context_object_queries = self.selfattn(x)[-1, :].unsqueeze(0)
                context_object_queries = x
                
                if self.continual_layers == None:
                    # test 1
                    x_query = self.individual_crossattn(x_query, context_object_queries)
                    x_support = self.individual_crossattn(x_support, context_object_queries) # supports 1 384
                else:
                    # test 2
                    x_query = self.continual_crossattn(x_query, context_object_queries)
                    x_support = self.continual_crossattn(x_support, context_object_queries)
                x_query = self.norm2(x_query) # 75 1 384
                x_support = self.norm2(x_support)
            
                prototypes = F.normalize(torch.mean(x_support.view(5, shots, self.ca_dim), dim=1), dim=-1) # 5 384
                x_query = x_query.transpose(0, 1) # objects queries dim
                prototypes = prototypes.unsqueeze(1).transpose(0, 1) # objects 5 dim
                
                x_query = F.normalize(x_query, dim=-1)
                distance = torch.einsum('oqd, owd -> oqw', x_query, prototypes).transpose(0, 1) # queries objects 5
                
                logits = distance.mean(dim=1)
                
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc