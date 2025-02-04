import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import copy

from models.SSL import MoCo, sinkhorn
from utils import split_support_query_set, mixup

from tqdm import tqdm


class PsCo(MoCo):
    def __init__(self, backbone, q_size=16384, momentum=0.99, mixup=False):
        super(PsCo, self).__init__(backbone, q_size, momentum)
        self.projector = self.make_mlp(input_dim=self.outdim, last_bn=True)
        self.predictor = self.make_mlp(input_dim=128, last_bn=False)
        self.shots = 4
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.mixup = mixup
    
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
                
        if self.mixup:
            x_q_mixup, labels_moco_aux, lam = mixup(x_q, alpha=1.)
            x_q_mixup = x_q_mixup.detach()
            q = F.normalize(self.predictor(self.projector(self.encoder(x_q_mixup))))
        else:
            q = F.normalize(self.predictor(self.projector(self.encoder(x_q))))
        
        #psco loss
        with torch.no_grad():
            k = F.normalize(self.projector_k(self.encoder_k(x_k))).detach()
            sim = torch.einsum('bd, qd -> bq', k, self.queue.clone().detach())
            labels_tilde = sinkhorn(sim, device)
            support_set, labels = self.select_topk(self.queue, labels_tilde)
            labels = labels.to(device)
        
        logits = torch.einsum('bd, sd -> bs', q, support_set)
        loss_psco = logits.logsumexp(dim=1) - (torch.sum(logits * labels, dim=1) / self.shots)
        loss_psco = loss_psco.mean()
        
        # moco loss
        if self.mixup:
            contrast = torch.cat([k, self.queue.clone().detach()], dim=0)
            logits = torch.einsum('bd, cd -> bc', q, contrast)
            labels_moco = torch.arange(x_q_mixup.shape[0], dtype=torch.long).cuda()
            loss_moco = torch.mean(lam * self.criterion(logits / 0.2, labels_moco) + (1. - lam) * self.criterion(logits / 0.2, labels_moco_aux))
        else:
            logits_moco_positive = torch.einsum('bd, bd -> b', q, k).unsqueeze(1) # b 1
            logits_moco_negative = torch.einsum('bd, qd -> bq', q, self.queue.clone().detach())
            logits_moco = torch.cat((logits_moco_positive, logits_moco_negative), dim=1) # b 1+q
            
            labels_moco = torch.zeros(batch_size).long().to(device)
            loss_moco = F.cross_entropy(logits_moco / 0.2, labels_moco)
        
        #enqueue
        self.enqueue(batch_size, k)
        
        return loss_psco + loss_moco
    
    def select_topk(self, queue, labels_tilde):
        _, indicies = torch.topk(labels_tilde, k=self.shots, dim=1)
        b, k = indicies.shape
        support_set = queue[indicies].clone().detach().view(b * k, -1) # b k d -> (bk) d
        
        labels = torch.zeros([b, b * k])
        
        for i in range(b):
            labels[i, k * i : k * i + k] = 1
        
        return support_set, labels
    
    def fewshot_acc(self, args, inputs, labels, device):
        with torch.no_grad():
            correct = 0
            total = 0
            
            x = self.projector(self.encoder(inputs))
            
            tasks = split_support_query_set(x, labels, device, num_tasks=1, num_shots=args.num_shots)
            
            for x_support, x_query, y_support, y_query in tasks:
                x_support = F.normalize(x_support)
                x_query = F.normalize(self.predictor(x_query)) # q d
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
            total = 0
            correct = 0
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            num_support = args.test_num_ways * args.num_shots
            A = torch.zeros(num_support, num_support, device=device)
            for i in range(args.test_num_ways):
                A[i * args.num_shots:(i + 1) * args.num_shots, i * args.num_shots:(i + 1) * args.num_shots] = 1.
            
            net = copy.deepcopy(self)
            net.eval()
            net.predictor.train()
            net.projector.train()
            
            optimizer = optim.SGD(list(net.projector.parameters()) + list(net.predictor.parameters()), lr=0.01, momentum=0.9, weight_decay=0.001)
            
            with torch.no_grad():
                inputs = net.encoder(inputs).detach()
            
            tasks = split_support_query_set(inputs, labels, device, num_tasks=1, num_shots=args.num_shots)
            
            for x_support, x_query, y_support, y_query in tasks:
                shots_train = x_support
                
                for _ in range(n_iters):
                    with torch.no_grad():
                        shots_train = shots_train.detach()
                    
                    z = net.projector(shots_train)
                    p = net.predictor(z)
                    z = F.normalize(z, dim=1)
                    p = F.normalize(p, dim=1)
                    
                    logits = torch.einsum('zd, pd -> zp', p, z) / 0.2
                    loss = (logits.logsumexp(dim=1) - logits.mul(A.clone().detach()).sum(dim=1).div(args.num_shots)).mean()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                net.eval()
                with torch.no_grad():
                    x_support = F.normalize(net.projector(x_support))
                    x_query = F.normalize(net.predictor(net.projector(x_query))) # q d
                    prototypes = F.normalize(torch.sum(x_support.view(5, args.num_shots, -1), dim=1), dim=1) # 5 d
                    
                    logits = torch.einsum('qd, wd -> qw', x_query, prototypes)
                    _, predicted = torch.max(logits.data, 1)
                    correct += (predicted == y_query).sum().item()
                    total += y_query.size(0)
                
                acc = 100 * correct / total
            
            total_acc += acc
        
        accuracy = total_acc / len(loader)
        return accuracy
