import torch
import torch.nn as nn

from functools import partial
import backbone.vision_transformer as vit


class ViTEncoder(vit.VisionTransformer):
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        add_cls_token=False,
        **kwargs):
        super().__init__(img_size, patch_size, in_chans, embed_dim, predictor_embed_dim, depth,
                         predictor_depth, num_heads, mlp_ratio, **kwargs)
    
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.add_cls_token = add_cls_token
        self.depth = depth
        num_patches = self.patch_embed.num_patches
        # --
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = vit.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=self.add_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        
    def forward_gprompt(self, x, split_layer, g_prompt=None):
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
        
        if self.add_cls_token:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        if g_prompt != None:
            g_prompt = g_prompt.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat((g_prompt, x), dim=1)
        
        # -- fwd prop
        for i in range(split_layer):
            x = self.blocks[i](x)
        
        return x
    
    def forward_eprompt(self, x, split_layer, e_prompt):
        e_prompt = e_prompt.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat((e_prompt, x), dim=1)
        
        for i in range(split_layer, self.depth):
            x = self.blocks[i](x)

        #if self.norm is not None:
        #    x = self.norm(x)
            
        return x
    
def vit_tiny(patch_size=16, **kwargs):
    model = ViTEncoder(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = ViTEncoder(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = ViTEncoder(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = ViTEncoder(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = ViTEncoder(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = ViTEncoder(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

