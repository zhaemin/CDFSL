import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim

import copy
from tqdm import tqdm
import numpy as np


import backbone.resnet_encoder as encoders
import backbone.vision_transformer as vit

from utils import split_support_query_set

class SSLFramework(nn.Module):
    def __init__(self, backbone, img_size=84, patch_size=6):
        super(SSLFramework, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.outdim = 256
        self.backbone = backbone
        self.encoder = self.make_encoder(self.backbone)
        self.projector = self.make_mlp(self.outdim)
        
    def make_mlp(self, input_dim, hidden_dim=2048, num_layers=2, out_dim=128, last_bn=False):
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
            )
        
        for i in range(num_layers - 2):
            mlp.append(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        if num_layers >= 2:
            mlp.append(nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ))
        
        if last_bn:
            mlp.append(nn.BatchNorm1d(out_dim))
        
        return mlp
    
    def make_encoder(self, backbone):
        if backbone == 'resnet10':
            encoder = encoders.ResNet10(num_classes=64, only_trunk=False, adaptive_pool=True)
            self.outdim = encoder.classifier.weight.shape[1] #512
            encoder.classifier = nn.Identity()
        elif backbone == 'conv5':
            encoder = encoders.convnet5()
            self.outdim = 256
        elif backbone == 'resnet18':
            encoder = encoders.ResNet18(num_classes=64, only_trunk=False)
            self.outdim = encoder.classifier.weight.shape[1]
            encoder.classifier = nn.Identity()
        elif backbone == 'vit_tiny':
            self.outdim = 192
            
            class Mean_dim_1(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return x.mean(dim=1)
            
            mean_layer = Mean_dim_1()
            encoder = nn.Sequential(
                vit.vit_tiny(img_size=[self.img_size], patch_size=self.patch_size),
                mean_layer)
        
        return encoder
    
    def encoding(self, x):
        return F.normalize(self.encoder(x), p=2)
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            
            x = self.encoder(inputs)
            
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

class SimCLR(SSLFramework):
    def __init__(self, backbone):
        super(SimCLR, self).__init__(backbone)
        
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        z1 = self.projector(self.encoder(inputs[0]))
        z2 = self.projector(self.encoder(inputs[1]))
        
        z1 = F.normalize(z1, p=2)
        z2 = F.normalize(z2, p=2)
        
        z = torch.cat((z1,z2), dim=0)
        logits = torch.einsum('ad, bd -> ab', z, z)
        logits.fill_diagonal_(float('-inf'))
        
        tmp_labels1 = torch.arange(batch_size, 2*batch_size)
        tmp_labels2 = torch.arange(0, batch_size)
        labels = torch.cat((tmp_labels1, tmp_labels2)).to(device)
        
        loss = F.cross_entropy(logits / 0.5, labels)
        
        return loss

class MoCo(SSLFramework):
    def __init__(self, backbone, q_size, momentum):
        super(MoCo, self).__init__(backbone)
        self.dim = 128
        
        self.register_buffer('queue', F.normalize(torch.randn(q_size, self.dim), p=2))
        self.register_buffer('idx', torch.zeros(1, dtype=torch.int64))
        self.max_queue_size = q_size
        self.momentum = momentum
        
        self.projector_k = self.make_mlp(self.outdim)
        self.encoder_k = self.make_encoder(backbone)
        
        for k_param, q_param in zip(self.encoder_k.parameters(), self.encoder.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
            
        for k_param, q_param in zip(self.projector_k.parameters(), self.projector.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False
        
    def forward(self, inputs, device):
        batch_size = inputs[0].size(0)
        
        x_q = inputs[0]
        x_k = inputs[1]
        
        #momentum update
        with torch.no_grad():
            for k_param, q_param in zip(self.encoder_k.parameters(), self.encoder.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
            
            for k_param, q_param in zip(self.projector_k.parameters(), self.projector.parameters()):
                k_param.data = self.momentum * k_param.data + (1 - self.momentum) * q_param.data
        
        q = F.normalize(self.projector(self.encoder(x_q)))
        k = F.normalize(self.projector_k(self.encoder_k(x_k)))
        k = k.detach() 
        
        queue = self.queue.clone().detach()
        logits_positive = torch.einsum('bd, bd -> b', q, k).unsqueeze(1)
        logits_negative = torch.einsum('bd, kd -> bk', q, queue)
        
        logits = torch.cat((logits_positive, logits_negative), dim=1)
        
        sim_labels = torch.zeros(batch_size).long().to(device)
        loss = F.cross_entropy(logits / 0.2, sim_labels)
        
        self.enqueue(batch_size, k)
            
        return loss
    
    def enqueue(self, batch_size, k):
        with torch.no_grad():
            idx = int(self.idx)
            self.queue[idx : idx + batch_size] = k
            self.idx[0] = (idx + batch_size) % self.max_queue_size

class SwAV(SSLFramework):
    def __init__(self, backbone):
        super(SwAV, self).__init__(backbone)
        self.temperature = 0.1
        self.dim = 128
        self.mm_prototypes = nn.Linear(128, 3000)
        
    def forward(self, inputs, device):
        with torch.no_grad():
            w = self.mm_prototypes.weight.data.clone()
            w = F.normalize(w, dim=1)
            self.mm_prototypes.weight.copy_(w)
        
        z1 = F.normalize(self.projector(self.encoder(inputs[0]))) # batchsize * 128
        z2 = F.normalize(self.projector(self.encoder(inputs[1])))
        
        scores_1 = self.mm_prototypes(z1)
        scores_2 = self.mm_prototypes(z2)
        
        with torch.no_grad():
            scores_1_tmp = scores_1.detach()
            scores_2_tmp = scores_2.detach()
            
            code_1 = sinkhorn(scores_1_tmp, device)
            code_2 = sinkhorn(scores_2_tmp, device)
            
        prob_1 = F.log_softmax(scores_1 / self.temperature, dim=1)
        prob_2 = F.log_softmax(scores_2 / self.temperature, dim=1)
        
        loss = -0.5 * torch.mean(torch.sum(code_1  * prob_2, dim=1) + torch.sum(code_2 * prob_1, dim=1))
        
        return loss
    
def sinkhorn(scores, device, eps=0.05, niters=3):
    code = torch.transpose(torch.exp(scores / eps), 0, 1)
    code /= torch.sum(code)
    
    k, b = code.shape
    u, r, c = torch.zeros(k), torch.ones(k) / k, torch.ones(b) / b
    u, r, c = u.to(device), r.to(device), c.to(device)
    
    for _ in range(niters):
        u = torch.sum(code, dim=1)
        code *= (r / u).unsqueeze(1) # k * b
        code *= (c / torch.sum(code, dim=0)).unsqueeze(0) # k * b
    
    return torch.transpose((code / torch.sum(code, dim=0, keepdim=True)), 0, 1) # b * k
