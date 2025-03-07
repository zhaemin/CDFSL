import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy
import math
import numpy as np


from utils import split_support_query_set

from backbone import vision_transformer as vit

from tqdm import tqdm

from models.ijepa import MIM


class MAE(MIM):
    def __init__(self, img_size, patch_size):
        super(MAE, self).__init__(img_size, patch_size)
        self.encoder, self.decoder = self.load_backbone(encoder_add_cls_token=True, pred_add_cls_token=True)
    
    def random_masking(self, batch_size, num_patches, mask_ratio, device):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = batch_size, num_patches
        len_keep = int(L * (1 - mask_ratio)) # 0.25 * L
        
        noise = torch.rand(N, L, device=device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove # 0 3 4 5 1 2
        ids_restore = torch.argsort(ids_shuffle, dim=1) # [0, 4, 5, 1, 2, 3] -> 원래 순서로 복원
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # N kL
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device) 
        mask[:, :len_keep] = 0 # 0 0 1 1 1 1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore) # masking된 부분은 1, unmasked 0으로 표시 -> 0 1 1 0 1 1
        
        return mask, ids_restore, ids_keep
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p)) # b 3 14 6 14 6
        x = torch.einsum('nchpwq->nhwpqc', x) # b 14 14 6 6 3
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3)) # b 14*14 36*3
        return x
    
    def forward(self, inputs, device):
        x = inputs
        mask, ids_restore, ids_keep = self.random_masking(x.size(0), self.num_patches, 0.75, device)
        
        x = self.encoder(x, ids_keep, return_attn=True)
        pred = self.decoder(x, ids_restore=ids_restore)
        
        target = self.patchify(inputs)
        
        # normalization
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss
    


    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            if args.model == 'ijepa':
                encoder = self.target_encoder
            else:
                encoder = self.encoder
                
            correct = 0
            total = 0
            
            #x = torch.mean(encoder(inputs), dim=1) # B K D -> B D
            #x = encoder(inputs)[:, 0, :]
            
            # attn weigted sum
            x = self.attn_weighted_sum(inputs, encoder)
            
            batchnorm = nn.BatchNorm1d(x.size(1), affine=False, eps=1e-6).to(device)
            x = batchnorm(x)
            
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
        #linear probing
        total_acc = 0
        
        for data in tqdm(loader, desc="Test ..."):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=args.num_shots)
            correct, total = 0, 0
            
            for x_support, x_query, y_support, y_query in tasks:
                if args.model == 'ijepa':
                    net = copy.deepcopy(self.target_encoder)
                else:
                    net = copy.deepcopy(self.encoder)
                    
                classifier = nn.Sequential(
                    nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6),
                    nn.Linear(self.encoder.embed_dim, args.train_num_ways)).to(device)
                optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
                
                net.eval()
                classifier.train()
                
                with torch.no_grad():
                    # global avg pooling
                    #shots   = torch.mean(net(x_support), dim=1)
                    #queries = torch.mean(net(x_query), dim=1)
                    
                    # attn_weighted sum
                    #shots = self.attn_weighted_sum(x_support, net)
                    #queries = self.attn_weighted_sum(x_query, net)
                    
                    # cls token
                    shots   = net(x_support)[:, 0, :]
                    queries = net(x_query)[:, 0, :]
                    
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



'''
    def ft_fewshot_acc(self, loader, device, n_iters, args):
        #linear probing
        total_acc = 0
        
        for data in tqdm(loader, desc="Test ..."):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=args.num_shots)
            correct, total = 0, 0
            
            for x_support, x_query, y_support, y_query in tasks:
                net = copy.deepcopy(self.encoder)
                weight_p = nn.Parameter(torch.randn(1, self.num_patches, 1, device=device))
                classifier = nn.Sequential(
                    nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6),
                    nn.Linear(self.encoder.embed_dim, args.train_num_ways)).to(device)
                optimizer = optim.SGD([weight_p]+list(classifier.parameters()), lr=0.01, momentum=0.9, weight_decay=0.001)
                
                net.eval()
                classifier.train()
                
                with torch.no_grad():
                    shots   = net(x_support)
                    queries = net(x_query)
                
                for _ in range(100):
                    with torch.no_grad():
                        shots   = shots.detach()
                        
                    rand_id = np.random.permutation(args.train_num_ways * args.num_shots)
                    batch_indices = [rand_id[i*4:(i+1)*4] for i in range(rand_id.size//4)]
                    for id in batch_indices:
                        x_train = shots[id]
                        y_train = y_support[id]
                        
                        weight_p = F.softmax(weight_p.detach(), dim=1) # F.softmax 적용해보기
                        weight_p_repeat = weight_p.repeat(x_train.size(0), 1, 1)
                        
                        x_train = torch.einsum('bkd, bpa -> bda', x_train, weight_p_repeat).squeeze()
                        shots_pred = classifier(x_train)
                        loss = F.cross_entropy(shots_pred, y_train)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                net.eval()
                classifier.eval()
                
                with torch.no_grad():
                    weight_p = F.softmax(weight_p.detach(), dim=1)
                    weight_p_repeat = weight_p.repeat(queries.size(0), 1, 1)
                    shots =  torch.einsum('bkd, bpa -> bda', shots, weight_p).squeeze()
                    queries = torch.einsum('bkd, bpa -> bda', queries, weight_p).squeeze()

                    shots = F.normalize(shots) # s d
                    queries = F.normalize(queries) # q d
                    prototypes = F.normalize(torch.sum(shots.view(5, args.num_shots, -1), dim=1), dim=1) # 5 d
                    
                    logits = torch.einsum('qd, wd -> qw', queries, prototypes)
                    _, predicted = torch.max(logits.data, 1)
                    correct += (predicted == y_query).sum().item()
                    total += y_query.size(0)
                    
            acc = 100 * correct / total
            total_acc += acc
            
        accuracy = total_acc / len(loader)
        return accuracy

'''