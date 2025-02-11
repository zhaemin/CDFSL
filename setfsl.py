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
        # binary로도 설정해보기 -> 이진수 변환
        numbers = torch.arange(0, 2**self.num_objects)
        self.binary = torch.tensor([[int(b) for b in f"{i:010b}"] for i in numbers], requires_grad=False).float().cuda() # 1024 10
        
        '''
        for param in self.encoder.parameters():
            param.requires_grad = False
        '''
        
    
    def forward(self, inputs, device):
        # inputs = [batchsize*2, 3, h, w]
        
        with torch.no_grad():
            z = self.encoder(inputs)
        z = self.decoder(self.query_pos.weight, z) # object bs*2
        z = z.transpose(0, 1) # bs*2 object
        x, y = torch.chunk(z, 2) # split to x, y
        x = F.normalize(x)
        y = F.normalize(y)
        
        # bs object
        p_x = F.log_softmax(x @ self.binary.T, dim=0) # bs 1024
        p_y = F.log_softmax(y @ self.binary.T, dim=0)
        
        loss = F.l1_loss(p_x, p_y)

        return loss

    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            encoder = self.encoder
                
            correct = 0
            total = 0
            
            x = torch.mean(encoder(inputs), dim=1) # B K D -> B D
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
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=args.num_shots)
            correct, total = 0, 0
            
            for x_support, x_query, y_support, y_query in tasks:
                net = copy.deepcopy(self.encoder)
                    
                classifier = nn.Sequential(
                    #nn.BatchNorm1d(self.encoder.embed_dim, affine=False, eps=1e-6),
                    nn.Linear(self.encoder.embed_dim*2, args.train_num_ways)).to(device)
                optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
                
                net.eval()
                classifier.train()
                
                with torch.no_grad():
                    # global avg pooling
                    #shots   = torch.mean(net(x_support), dim=1)
                    #queries = torch.mean(net(x_query), dim=1)
                    
                    # attn_weighted sum
                    shots = self.attn_weighted_sum(x_support, net)
                    queries = self.attn_weighted_sum(x_query, net)
                    
                    # cls token
                    #shots   = net(x_support)[:, 0, :]
                    #queries = net(x_query)[:, 0, :]
                    
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
