import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_support_query_set

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        return x



class ProtoNet(nn.Module):
    def __init__(self, backbone=ConvNet(), emb_dim=64):
        super(ProtoNet, self).__init__()
        self.embedding = backbone
        self.emb_dim = emb_dim
        
    def forward(self, args, inputs, labels, num_ways, device):
        inputs = self.embedding(inputs)
        tasks = split_support_query_set(inputs, labels, device, num_tasks=4, num_shots=args.num_shots)
            
        total_loss = 0
        
        for task in tasks:
            support_set, query_set, y_support, y_query = task
            
            #for prediction
            prototypes = torch.mean(support_set.view(num_ways, -1, self.emb_dim), dim=1)
            logits = -torch.cdist(query_set, prototypes)
            
            loss = F.cross_entropy(logits, y_query)
            total_loss += loss
            
        acc = None
        if not self.training:
            with torch.no_grad():
                _, predicted = torch.max(logits.data,1)
                total = args.num_queries * args.test_num_ways
                correct = (predicted == y_query).sum().item()
                acc = 100*correct/total
                
        return acc
