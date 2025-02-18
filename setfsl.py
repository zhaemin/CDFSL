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
from models.mae import MIM


class SET_FSL(MIM):
    def __init__(self, img_size, patch_size):
        super(SET_FSL, self).__init__(img_size, patch_size)
        self.add_cls_token = True
        self.encoder, self.decoder = self.load_backbone(encoder_add_cls_token=self.add_cls_token, pred_add_cls_token=self.add_cls_token, setfsl=True)
        self.outdim = self.encoder.embed_dim
        self.num_objects = 10
        
        self.query_pos = nn.Embedding(self.num_objects, self.encoder.embed_dim)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        
    def forward(self, inputs, labels, device):
        # inputs = [batchsize*2, 3, h, w]
        
        with torch.no_grad():
            z, encoder_attn = self.encoder(inputs, return_attn=True)
        encoder_attn = F.normalize(encoder_attn.mean(axis = 1)[:, 0, 1:]).unsqueeze(dim=-2) #bs 1 numpatches
        z, decoder_attn = self.decoder(self.query_pos.weight, z) # object samples dim / attn(query의 각 object가 어디에 집중) -> bs object patches+1
        
        decoder_attn = F.normalize(decoder_attn.mean(axis = 1)[:, 1:]).unsqueeze(dim=-1) # object들의 mean을 구함, cls token 제외 -> bs numpatches 1
        attn_dist = torch.einsum('bep, bpd -> bed', encoder_attn, decoder_attn).mean()
        
        z = z.transpose(0, 1) # samples object dim
        tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
        
        loss = 0
        for x_support, x_query, y_support, y_query in tasks:
            prototypes = torch.mean(x_support.view(5, -1, self.num_objects, self.outdim), dim=1)
            prototypes = F.normalize(prototypes, dim=1)
            x_query = F.normalize(x_query, dim=1)
            
            x_query = x_query.unsqueeze(1).repeat(1, prototypes.size(0), 1, 1).unsqueeze(-2) # queries num_ways objects 1 dim
            prototypes = prototypes.unsqueeze(0).repeat(x_query.size(0), 1, 1, 1).unsqueeze(-1) # queries num_ways objects dim 1
            distance = (x_query @ prototypes).squeeze() # queries num_ways objects
            #distance = torch.einsum('qad, pbd -> qpab', x_query, prototypes).reshape(x_query.size(0), 5, -1)# 75 5 10 10
            logits = torch.sum(distance, dim=-1) # queries num_ways
            
            loss += F.cross_entropy(logits, y_query)
            
        return 0.2 * loss + 0.8 * attn_dist
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            
            z, encoder_attn = self.encoder(inputs, return_attn=True)
            cls_token = z[:, 0, :].unsqueeze(1)
            z, decoder_attn = self.decoder(self.query_pos.weight, z) # object samples dim / attn(query의 각 object가 어디에 집중) -> bs object patches+1
            
            z = z.transpose(0, 1) # samples object dim
            z = torch.concat((cls_token, z), dim=1)
            tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=5)
            
            for x_support, x_query, y_support, y_query in tasks:
                prototypes = torch.mean(x_support.view(5, -1, self.num_objects+1, self.outdim), dim=1)
                prototypes = F.normalize(prototypes, dim=1)
                x_query = F.normalize(x_query, dim=1)
                
                x_query = x_query.unsqueeze(1).repeat(1, prototypes.size(0), 1, 1).unsqueeze(-2) # queries num_ways objects 1 dim
                prototypes = prototypes.unsqueeze(0).repeat(x_query.size(0), 1, 1, 1).unsqueeze(-1) # queries num_ways objects dim 1
                distance = (x_query @ prototypes).squeeze() # queries num_ways objects
                
                cls_distance = distance[:, :, 0]
                object_distance = distance[:, :, 1:].sum(dim=-1)
                #distance = torch.einsum('qad, pbd -> qpab', x_query, prototypes).reshape(x_query.size(0), 5, -1) # 75 5 10 10
                #logits = 0.5 * cls_distance + 0.5 * object_distance
                logits = object_distance
                
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
            
            with torch.no_grad():
                z, encoder_attn = self.encoder(inputs, return_attn=True)
                z, decoder_attn = self.decoder(self.query_pos.weight, z) # object samples dim / attn(query의 각 object가 어디에 집중) -> bs object patches+1
                
                z = z.transpose(0, 1) # samples object dim
                z = z.sum(dim=1)
            
            tasks = split_support_query_set(z, labels, device, num_tasks=1, num_shots=args.num_shots)
            correct, total = 0, 0
            
            for x_support, x_query, y_support, y_query in tasks:
                #net = copy.deepcopy(self.encoder)
                    
                classifier = nn.Sequential(
                    #nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6),
                    nn.Linear(self.encoder.embed_dim, args.train_num_ways)).to(device)
                optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
                
                self.encoder.eval()
                self.decoder.eval()
                classifier.train()
                    
                for _ in range(100):
                    with torch.no_grad():
                        shots   = x_support.detach()
                        queries = x_query.detach()
                        
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
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
                
            correct = 0
            total = 0
            
            x = self.encoder(inputs)
            x = self.decoder(self.query_pos.weight, x).transpose(0, 1)
            x = F.normalize(x)
            x = F.log_softmax(x @ self.binary.T, dim=0)
            #x = encoder(inputs)[:, 0, :] # use cls token
            #batchnorm = nn.BatchNorm1d(x.size(1), affine=False, eps=1e-6).to(device)
            #x = batchnorm(x)
            
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
            
            correct, total = 0, 0
            
            encoder = copy.deepcopy(self.encoder)
            decoder = copy.deepcopy(self.decoder)
            
            optimizer = optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
            
            encoder.eval()
            decoder.train()
            
            with torch.no_grad():
                inputs = encoder(inputs)
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=args.num_shots)
                
            for x_support, x_query, y_support, y_query in tasks:
                for _ in range(n_iters):
                    shots = decoder(x_support)
                    queries = decoder(x_query)
                    
                    prototypes = F.normalize(torch.sum(shots.view(5, args.num_shots, -1), dim=1), dim=1) # 5 d
                    logits = torch.einsum('qd, wd -> qw', queries, prototypes)
                    
                    loss = F.cross_entropy(logits, y_query)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    x_support = F.normalize(x_support)
                    x_query = F.normalize(x_query) # q d
                    prototypes = F.normalize(torch.sum(x_support.view(5, args.num_shots, -1), dim=1), dim=1) # 5 d
                    
                    logits = torch.einsum('qd, wd -> qw', x_query, prototypes)
                    _, predicted = torch.max(logits.data, 1)
                    correct += (predicted == y_query).sum().item()
                    total += y_query.size(0)
                    acc = 100 * correct / total
            
            total_acc += acc
        
        accuracy = total_acc / len(loader)
        return accuracy
    
    '''
