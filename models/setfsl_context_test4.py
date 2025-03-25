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
        self.init_o()
        
    def init_o(self):
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
    
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
        
        self.norm = nn.LayerNorm(self.encoder_dim)
        
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
        kv = z
        x = self.crossattn(q, kv)
        x = q + x
        
        return x
    
    def continual_crossattn(self, z, object_queries):
        x = object_queries
        
        for blk, l in zip(self.ca_blocks, self.layer_nums):
            q = x
            kv = z[l]
            x = blk(q, kv)
            
            x = x + q
            
        return x

    def attn_weighted_sum(self, patch_tokens, attn):
        attn_weights = attn.mean(axis = 1)[:, 0, 1:] # head별 mean 계산 후 cls token 분리(cls token 제외 나머지 token들에 대해) B 1 num_patches
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
        weighted_patch_tokens = torch.sum(patch_tokens * attn_weights.unsqueeze(-1), dim=1) # B N D * B N 1-> B D
        
        return weighted_patch_tokens
    
    def forward(self, inputs, labels, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=5)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            x_support, support_attn = self.encoder(x_support, return_attn=True, memories=True)
            x_query, query_attn = self.encoder(x_query, return_attn=True, memories=True)
            
            shots = x_support.size(1) // 5
            x_support_patches = x_support[self.layer][:, 1:, :].reshape(-1, self.encoder_dim) # 25*196 384
            #prototypes_patches = F.normalize(torch.mean(x_support_patches.view(5, shots, -1, self.encoder_dim), dim=1), dim=-1).reshape(-1, self.encoder_dim) # 5*196 384
            
            x_query = x_query[self.layer] # queries 197 384
            x_support_patches = x_support_patches.repeat(x_query.size(0), 1, 1) # queries 25*196 384
            
            if self.continual_layers == None:
                # test 1
                contextualized_x = self.individual_crossattn(x_query, x_support_patches)
            else:
                # test 2
                contextualized_x  = self.continual_crossattn(x_query, x_support_patches)
            x_support_patches = self.norm(contextualized_x) # queries 5*196 384
            x_support_patches = x_support_patches.reshape(-1, self.num_patches, self.encoder_dim)  # queries*5 196 384
            x_query = self.norm(x_query)
            
            support_attn = support_attn.repeat(x_query.size(0), 1, 1, 1) # queries*5 6 197 197
            x_support_patches = self.attn_weighted_sum(x_support_patches, support_attn) # queries*supports 384
            prototypes = torch.mean(x_support_patches.view(x_query.size(0), -1, shots, self.encoder_dim), dim=2) # queries 5 384
            x_query = self.attn_weighted_sum(x_query[:, 1:, :], query_attn)
            
            prototypes = F.normalize(prototypes, dim=-1) # 75 5 384
            x_query = F.normalize(x_query, dim=-1).unsqueeze(1) # 75 1 384
            
            distance = torch.einsum('bqd, bwd -> bqw', x_query, prototypes) # 75 5
            
            logits = (distance / self.temperature).reshape(-1, 5)
            loss += F.cross_entropy(logits, y_query)
            
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            loss = 0
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=5)
            
            for x_support, x_query, y_support, y_query in tasks:
                x_support, support_attn = self.encoder(x_support, return_attn=True, memories=True)
                x_query, query_attn = self.encoder(x_query, return_attn=True, memories=True)
                
                shots = x_support.size(1) // 5
                x_support_patches = x_support[self.layer][:, 1:, :].reshape(-1, self.encoder_dim) # 25*196 384
                #prototypes_patches = F.normalize(torch.mean(x_support_patches.view(5, shots, -1, self.encoder_dim), dim=1), dim=-1).reshape(-1, self.encoder_dim) # 5*196 384
                
                x_query = x_query[self.layer] # queries 197 384
                x_support_patches = x_support_patches.repeat(x_query.size(0), 1, 1) # queries 25*196 384
                
                if self.continual_layers == None:
                    # test 1
                    contextualized_x = self.individual_crossattn(x_query, x_support_patches)
                else:
                    # test 2
                    contextualized_x  = self.continual_crossattn(x_query, x_support_patches)
                x_support_patches = self.norm(contextualized_x) # queries 5*196 384
                x_support_patches = x_support_patches.reshape(-1, self.num_patches, self.encoder_dim)  # queries*5 196 384
                x_query = self.norm(x_query)
                
                support_attn = support_attn.repeat(x_query.size(0), 1, 1, 1) # queries*5 6 197 197
                x_support_patches = self.attn_weighted_sum(x_support_patches, support_attn) # queries*supports 384
                prototypes = torch.mean(x_support_patches.view(x_query.size(0), -1, shots, self.encoder_dim), dim=2) # queries 5 384
                x_query = self.attn_weighted_sum(x_query[:, 1:, :], query_attn)
                
                prototypes = F.normalize(prototypes, dim=-1) # 75 5 384
                x_query = F.normalize(x_query, dim=-1).unsqueeze(1) # 75 1 384
                
                distance = torch.einsum('bqd, bwd -> bqw', x_query, prototypes) # 75 5
                
                logits = (distance / self.temperature).reshape(-1, 5)
                loss += F.cross_entropy(logits, y_query)
                
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc

    def return_prototypes(self, inputs, labels, device):
        tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=5, num_queries=1)
        
        for x_support, x_query, y_support, y_query in tasks:
            x_support, support_attn = self.encoder(x_support, return_attn=True, memories=True)
            x_query, query_attn = self.encoder(x_query, return_attn=True, memories=True)
            shots = x_support.size(1) // 5
            x_support_patches = x_support[self.layer][:, 1:, :].reshape(-1, self.encoder_dim) # 25*196 384
            #prototypes_patches = F.normalize(torch.mean(x_support_patches.view(5, shots, -1, self.encoder_dim), dim=1), dim=-1).reshape(-1, self.encoder_dim) # 5*196 384
            
            x_query = x_query[self.layer] # queries 197 384
            x_support_patches = x_support_patches.repeat(x_query.size(0), 1, 1) # queries 25*196 384
            
            if self.continual_layers == None:
                # test 1
                contextualized_x = self.individual_crossattn(x_query, x_support_patches)
            else:
                # test 2
                contextualized_x  = self.continual_crossattn(x_query, x_support_patches)
            x_support_patches = self.norm(contextualized_x) # queries 5*196 384
            x_support_patches = x_support_patches.reshape(-1, self.num_patches, self.encoder_dim)  # queries*5 196 384
            x_query = self.norm(x_query)
            
            support_attn = support_attn.repeat(x_query.size(0), 1, 1, 1) # queries*5 6 197 197
            x_support_patches = self.attn_weighted_sum(x_support_patches, support_attn) # queries*supports 384
            prototypes = torch.mean(x_support_patches.view(x_query.size(0), -1, shots, self.encoder_dim), dim=2) # queries 5 384
            x_query = self.attn_weighted_sum(x_query[:, 1:, :], query_attn)
            
            prototypes = F.normalize(prototypes, dim=-1) # 75 5 384
            x_query = F.normalize(x_query, dim=-1).unsqueeze(1) # 75 1 384
            
            prototypes = prototypes[0] # 5 384
            x_query = x_query[0] # 1 384
            y_query = y_query[0]
            
        return prototypes, x_query, y_query