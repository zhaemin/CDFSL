# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
import numpy as np

import torch
import torch.nn as nn


from models.vit_utils import trunc_normal_, repeat_interleave_batch, apply_masks
from backbone.pos_embed import get_2d_sincos_pos_embed


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvEmbed(nn.Module):
    """
    3x3 Convolution stems for ViT following ViTC models
    """

    def __init__(self, channels, strides, img_size=224, in_chans=3, batch_norm=True):
        super().__init__()
        # Build the stems
        stem = []
        channels = [in_chans] + channels
        for i in range(len(channels) - 2):
            stem += [nn.Conv2d(channels[i], channels[i+1], kernel_size=3,
                            stride=strides[i], padding=1, bias=(not batch_norm))]
            if batch_norm:
                stem += [nn.BatchNorm2d(channels[i+1])]
            stem += [nn.ReLU(inplace=True)]
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])]
        self.stem = nn.Sequential(*stem)

        # Comptute the number of patches
        stride_prod = int(np.prod(strides))
        self.num_patches = (img_size[0] // stride_prod)**2

    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)


class Attention_QKV(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
    
        self.attn_drop = nn.Dropout(attn_drop)
        
        #self.proj = nn.Linear(dim, dim, bias=False) # 이 projection weight 0으로 만들기
        self.proj = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        #self.init_proj_weight()
        
        #self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #self.k = nn.Linear(dim, dim, bias=qkv_bias)
        #self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.q = nn.Identity()
        self.k = nn.Identity()
        self.v = nn.Identity()
    
    def init_proj_weight(self):
        nn.init.zeros_(self.proj.weight)
        '''
        for param in self.proj.parameters():
            param.requires_grad = False
        '''
        
    def forward(self, q, k, v):
        B, N, C = q.shape
        
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale # bs object p
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        #self.init_proj_weight()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class VisionTransformerDecoder(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads=1):
        super().__init__()
        self.attention = Attention_QKV(embed_dim, num_heads)
        self.norm  = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, embed_dim*4)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.weight_for_blk = nn.Parameter(torch.ones(12, requires_grad=True))
        
        
    def forward(self, query_pos, memories, attn, layer):
        bs = memories[0].size(0)
        num_objects = query_pos.size(0)
        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1) # bs object dim
        object_queries = torch.zeros_like(query_pos)
        object_queries = object_queries + query_pos
        
        cls_token = memories[layer][:, 0, :].unsqueeze(1)
        x1 = torch.concat((object_queries, cls_token), dim=1)
        
        if attn == 'selfattn':
            # self - attention
            x2 = self.attention(x1, x1, x1)
        else:
            # cross - attention
            q = query_pos
            k = memories[layer]
            v = memories[layer]
            
            x2 = self.attention(q, k, v)
            x2 = torch.concat((x2, torch.zeros_like(cls_token)), dim=1)
        
        x  = x1 + x2
        
        x = self.norm(x[:, :num_objects+1, :]) # only cls token + object tokens
        #x = self.norm(x[:, num_objects, :]).unsqueeze(1) # only cls token
        #x = self.norm(x[:, :num_objects, :]) # only obj
        
        return x

'''
    def forward(self, query_pos, memories, attn, layer):
        bs = memories[0].size(0)
        num_objects = query_pos.size(0)
        query_pos = query_pos.unsqueeze(0).repeat(bs, 1, 1) # bs object dim
        object_queries = torch.zeros_like(query_pos)
        object_queries = object_queries + query_pos
        
        cls_token_shape = torch.zeros_like(memories[layer][:, 0, :].unsqueeze(1))
        #x1 = torch.concat((object_queries, cls_token), dim=1)
        
        layers = [2, 5, 8, 11]
        for i, l in enumerate(layers):
            # cross - attention
            q = query_pos
            k = memories[l]
            v = memories[l]
            
            x = self.attention(q, k, v)
            query_pos = query_pos + x
            
            if i == len(layers)-1:
                x = torch.concat((x, memories[l][:, 0, :].unsqueeze(1)), dim=1)
        
        x = self.norm(x[:, :num_objects+1, :]) # only cls token + object tokens
        
        return x
'''

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        add_class_token=False,
        patch_size=6,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.add_cls_token = add_class_token
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # ------
        if add_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=self.add_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.decoder_proj = nn.Linear(predictor_embed_dim, patch_size ** 2 * 3, bias=True)
        # --
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        # --
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, masks_x=None, masks=None, ids_restore=None):
        if ids_restore == None:
            return self.forward_ijepa_predictor(x, masks_x, masks)
        else:
            return self.forward_mae_decoder(x, ids_restore)
    
    def forward_mae_decoder(self, x, ids_restore):
        # embed tokens
        x = self.predictor_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) # 1 1 pd -> b (N+1-unmasked) 1
        if self.add_cls_token:
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle -> 원래 patch 위치로 mask token 이동
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x_ = torch.cat([x, mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) 
        
        # add pos embed
        x = x + self.pos_embed
        
        # apply Transformer blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)
        
        # predictor projection
        x = self.decoder_proj(x)
        
        # remove cls token
        if self.add_cls_token:
            x = x[:, 1:, :]
        
        return x

    def forward_ijepa_predictor(self, x, masks_x, masks): # x=encoder를 통과한 context, masks_x = mask_enc, masks = mask_pred
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'
        
        if self.add_cls_token:
            x = x[:, 1:, :] # cls token 제외
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
            
        if not isinstance(masks, list):
            masks = [masks]
            
        # -- Batch Size
        B = len(x) // len(masks_x)
        
        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x) # B K D
        
        # -- add positional embedding to x tokens
        x_pos_embed = self.pos_embed.repeat(B, 1, 1) #B N D
        x += apply_masks(x_pos_embed, masks_x) # context 부분의 position embedding만 남겨서 더함 -> num_context * B, K, D
        
        _, N_ctxt, D = x.shape
        
        # -- concat mask tokens to x
        pos_embs = self.pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks) # predict할 부분의 position embedding -> num_predict * B, K_pred, D
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) #context 수 만큼 반복, num_context * num_predict * B, K_pred, D
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1) # num_context * num_predict * B, K_pred, D
        # --
        pred_tokens += pos_embs #mask token에 predict mask의 position embedding을 더함
        x = x.repeat(len(masks), 1, 1) # num_predict * num_context * B, K, D
        x = torch.cat([x, pred_tokens], dim=1) # a+1, a+2, a+3 ...
        
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)
        
        # -- return preds for mask tokens
        x = x[:, N_ctxt:] # pred token만 분리
        x = self.predictor_proj(x)
        
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        add_cls_token=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.add_cls_token = add_cls_token
        self.depth = depth
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=self.add_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        
        # feature aggregation
        #self.blk_mix = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()
    

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
            
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None, return_attn=False, find_optimal_target=False):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]
        
        # -- patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # -- add positional embedding to x
        if self.add_cls_token:
            pos_embed = self.pos_embed
            x = x + pos_embed[:, 1:, :]
        else:
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
            x = x + pos_embed
        
        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)
        
        if self.add_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # -- fwd prop
        attn = None
        feature_lst = []
        
        for i, blk in enumerate(self.blocks):
            if i == self.depth-1: # last layer
                if return_attn:
                    x, attn = blk(x, return_attention=return_attn)
                else:
                    x = blk(x)
            else:
                x = blk(x)
            if find_optimal_target:
                feature_lst.append(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if find_optimal_target:
            feature_lst = torch.stack(feature_lst)
            return feature_lst
        elif return_attn:
            return x, attn
        else:
            return x

    def interpolate_pos_encoding(self, x, pos_embed): # 인수로 들어온 num patches(position embedding)과 실제 num patches가 다를 때 pos_embed를 interpolate
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model

def vit_decoder(num_patches, embed_dim, num_heads):
    model = VisionTransformerDecoder(num_patches, embed_dim, num_heads)
    return model

def vit_mini(patch_size=16, **kwargs):
    model = VisionTransformer(
        #embed_dim=192
        patch_size=patch_size, embed_dim=96, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        #embed_dim=192
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        # change embed dim -> original : 1024
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        #org: 1280 
        patch_size=patch_size, embed_dim=96, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}