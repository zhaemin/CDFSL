import torch
import torch.nn as nn

from functools import partial
import backbone.vision_transformer as vit


class ViTPredictor(vit.VisionTransformerPredictor):
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        add_cls_token=False,
        patch_size=6,
        **kwargs
        ):
        super().__init__(num_patches, embed_dim, predictor_embed_dim, depth, num_heads, mlp_ratio)
        
        self.add_cls_token = add_cls_token
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        pos_embed = vit.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=self.add_cls_token)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        self.decoder_proj = nn.Linear(predictor_embed_dim, patch_size ** 2 * 3, bias=True)
        
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
        x += vit.apply_masks(x_pos_embed, masks_x) # context 부분의 position embedding만 남겨서 더함 -> num_context * B, K, D
        
        _, N_ctxt, D = x.shape
        
        # -- concat mask tokens to x
        pos_embs = self.pos_embed.repeat(B, 1, 1)
        pos_embs = vit.apply_masks(pos_embs, masks) # predict할 부분의 position embedding -> num_predict * B, K_pred, D
        pos_embs = vit.repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) #context 수 만큼 반복, num_context * num_predict * B, K_pred, D
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


def vit_predictor(**kwargs):
    model = ViTPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
