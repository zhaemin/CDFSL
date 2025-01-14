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

from models.ijepa_utils import repeat_interleave_batch, apply_masks


class I_JEPA(nn.Module):
    def __init__(self, num_epochs):
        super(I_JEPA, self).__init__()
        self.encoder, self.predictor = self.load_backbone()
        self.target_encoder = copy.deepcopy(self.encoder)
        self.ipe = 9600
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.momentum_scheduler = (0.996 + i * (1.0 - 0.996) / (self.ipe * num_epochs)
                        for i in range(int(self.ipe * num_epochs) + 1))
    
    def load_backbone(backbone):
        encoder = vit.__dict__['vit_tiny'](img_size=[84], patch_size=8) #embed_dim=192, num_head=6
        predictor = vit.__dict__['vit_predictor'](num_patches=10 * 10, embed_dim=encoder.embed_dim, predictor_embed_dim=96)
        
        return encoder, predictor
    
    def forward(self, inputs, device):
        udata, masks_enc, masks_pred = inputs[0], inputs[1], inputs[2]
        
        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(device, non_blocking=True)
            masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
            return (imgs, masks_1, masks_2)
        
        imgs, masks_enc, masks_pred = load_imgs()
        
        def forward_target():
            with torch.no_grad():
                h = self.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                B = len(h)
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred) #prediction할 부분 -> num_pred * B, K, D
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc)) # num_context * num_pred * B, K, D
                return h
            
        def forward_context():
            z = self.encoder(imgs, masks_enc)
            z = self.predictor(z, masks_enc, masks_pred)
            return z
        
        h = forward_target()
        z = forward_context()
        loss = F.smooth_l1_loss(z, h)
        #distance = torch.pairwise_distance(h, z).view(-1, len(masks_enc) * len(masks_pred), h.size(1)) # 256 4 K
        #loss = torch.mean(torch.sum(distance, dim=-1))
        
        # Step 3. momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        
        return loss
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            
            x = self.encoder(inputs) # B K D
            x = x[:, 0, :].squeeze()
            
            tasks = split_support_query_set(x, labels, device, num_tasks=1, num_shots=args.num_shots)
            
            for x_support, x_query, y_support, y_query in tasks:
                x_support = F.normalize(x_support)
                x_query = F.normalize(x_query) # q d
                prototypes = F.normalize(torch.sum(x_support.view(5, args.num_shots, -1), dim=1), dim=1) # 5 d
                
                logits = torch.einsum('qd, wd -> qw', x_query, prototypes)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y_query).sum().item()
                total += y_query.size(0)
                
            acc = 100 * correct / total
        return acc
    
    def ft_fewshot_acc(self, loader, device, n_iters, args):
        total_acc = 0
        
        for data in tqdm(loader, desc="Test ..."):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=args.num_shots)
            correct, total = 0, 0
            
            for x_support, x_query, y_support, y_query in tasks:
                net = copy.deepcopy(self.encoder)
                classifier = nn.Linear(self.outdim, args.train_num_ways).to(device)
                optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
                
                net.eval()
                classifier.train()
                
                with torch.no_grad():
                    shots   = net(x_support)
                    queries = net(x_query)
                        
                for _ in range(100):
                    with torch.no_grad():
                        shots   = shots.detach()
                        queries = queries.detach()
                        
                    rand_id = np.random.permutation(args.train_num_ways * args.num_shots)
                    batch_indices = [rand_id[i*4:(i+1)*4] for i in range(rand_id.size//4)]
                    for id in batch_indices:
                        x_train = shots[id]
                        y_train = y_support[id]
                        shots_pred = classifier(x_train)
                        loss = F.cross_entropy(shots_pred, y_train)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                net.eval()
                classifier.eval()
                
                with torch.no_grad():
                    logits = classifier(queries)
                    _, predicted = torch.max(logits.data, 1)
                    correct += (predicted == y_query).sum().item()
                    total += y_query.size(0)
                    
            acc = 100 * correct / total
            total_acc += acc
            
        accuracy = total_acc / len(loader)
        return accuracy