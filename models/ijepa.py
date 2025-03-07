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
import backbone.vit_predictor as vit_predictor

from tqdm import tqdm

from backbone.vit_utils import repeat_interleave_batch, apply_masks

class MIM(nn.Module):
    def __init__(self, img_size, patch_size):
        super(MIM, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
    
    def load_backbone(self, encoder_add_cls_token=False, pred_add_cls_token=False, setfsl=False):
        print('img_size: ',self.img_size)
        print('patch_size: ',self.patch_size)
        print('num_patches: ',self.num_patches)
        encoder = vit_encoder.__dict__['vit_small'](img_size=[self.img_size], patch_size=self.patch_size, add_cls_token=encoder_add_cls_token)
        predictor = vit_predictor.__dict__['vit_predictor'](patch_size=self.patch_size, num_patches= (self.img_size//self.patch_size) ** 2, embed_dim=encoder.embed_dim, 
                                                predictor_embed_dim=encoder.embed_dim//2, num_heads=encoder.num_heads, add_cls_token=pred_add_cls_token)
        
        return encoder, predictor
    
    def attn_weighted_sum(self, x, encoder):
        x, attn = encoder(x, return_attn=True) # use cls token
        cls_token = x[:, 0, :]
        patch_tokens = x[:, 1:, :]
        attn_weights = attn.mean(axis = 1)[:, 0, 1:] # head별 mean 계산 후 cls token 분리(cls token 제외 나머지 token들에 대해) B 1 num_patches
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)
        weighted_patch_tokens = torch.sum(patch_tokens * attn_weights.unsqueeze(-1), dim=1) # B N D * B N -> B D
        
        x = torch.cat((cls_token, weighted_patch_tokens), dim=-1)
        x = F.normalize(x, dim=-1)
        
        return x

class I_JEPA(MIM):
    def __init__(self, img_size, patch_size, num_epochs):
        super(I_JEPA, self).__init__(img_size, patch_size)
        self.add_cls_token = True
        self.encoder, self.predictor = self.load_backbone(encoder_add_cls_token=self.add_cls_token, pred_add_cls_token=False)
        self.outdim = self.encoder.embed_dim
        self.target_encoder = copy.deepcopy(self.encoder)
        self.ipe = 150
        
        self.feature_weight = nn.Parameter(torch.randn(self.encoder.depth))
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.momentum_scheduler = (0.999 + i * (1.0 - 0.999) / (self.ipe * num_epochs)
                        for i in range(int(self.ipe * num_epochs) + 1))
    
    def forward(self, inputs, device):
        udata, masks_enc, masks_pred = inputs[0], inputs[1], inputs[2]
        
        # momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            #m = 0.999
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        
        
        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(device, non_blocking=True)
            masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
            return (imgs, masks_1, masks_2)
        
        imgs, masks_enc, masks_pred = load_imgs()
        
        def forward_target():
            with torch.no_grad():
                h = self.target_encoder(imgs, return_attn=False) # B num_patches D
                
            with torch.no_grad():
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim => patch 별
                B = len(h)
                # -- create targets (masked regions of h)
                if self.add_cls_token:
                    target_cls_token = h[:, 0, :]
                    h = h[:, 1:, :]
                h = apply_masks(h, masks_pred) #prediction할 부분 -> num_pred * B, K, D
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc)) # num_context * num_pred * B, K, D
                return h, target_cls_token
            
        def forward_context():
            z = self.encoder(imgs, masks_enc)
            if self.add_cls_token:
                context_cls_token = z[:, 0, :]
                z = z[:, 1:, :]
            z = self.predictor(z, masks_enc, masks_pred)
            return z, context_cls_token
        
        h, target_cls_token = forward_target()
        z, context_cls_token = forward_context()
        cls_loss = F.smooth_l1_loss(target_cls_token, context_cls_token)
        loss = F.smooth_l1_loss(z, h) + cls_loss

        #distance = torch.pairwise_distance(h, z).view(-1, len(masks_enc) * len(masks_pred), h.size(1)) # batchsize 4 K
        #loss = torch.mean(torch.sum(distance, dim=-1))
        
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            if args.model == 'ijepa':
                encoder = self.target_encoder
            else:
                encoder = self.encoder
                
            correct = 0
            total = 0
            
            x = torch.mean(encoder(inputs), dim=1) # B K D -> B D
            #x = self.attn_weighted_sum(inputs, encoder)
            #x = encoder(inputs)[:, 0, :]
            
            tasks = split_support_query_set(x, labels, device, num_tasks=1, num_shots=args.num_shots)
            
            for x_support, x_query, y_support, y_query in tasks:
                x_query = F.normalize(x_query, dim=-1) # q d
                prototypes = F.normalize(torch.mean(x_support.view(5, args.num_shots, -1), dim=1), dim=-1) # 5 d
                
                logits = torch.einsum('qd, wd -> qw', x_query, prototypes)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc