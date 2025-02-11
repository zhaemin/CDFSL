import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Sampler, DataLoader

from models.multiblock import MaskCollator as MBMaskCollator

import numpy as np
import cd_dataset

class SSLTransform(torch.nn.Module):
    def __init__(self, img_size, model):
        super(SSLTransform, self).__init__()
        self.model = model
        self.transform_strong = transforms.Compose([ 
            transforms.RandomResizedCrop(img_size, scale=(0.2 ,1)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        self.transform_weak = transforms.Compose([ 
            transforms.RandomResizedCrop(img_size, scale=(0.2 ,1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        self.transform_test = transforms.Compose([ 
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
    def __call__(self, x):
        x1 = self.transform_strong(x)
        if self.model == 'psco':
            x2 = self.transform_weak(x)
        else:
            x2 = self.transform_strong(x)
        x = self.transform_test(x)
        
        return [x1, x2, x]

class FewShotSampler(Sampler):
    def __init__(self, labels, num_ways, num_shots, num_queries, episodes, num_tasks, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.episodes = episodes
        self.num_tasks = num_tasks
        
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.data_matrix = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int)*np.nan)
        self.num_per_class = torch.zeros_like(self.classes)
        
        #data_matrix => 해당 class에 맞는 데이터의 index를 저장
        #np.where => nan인 값들이 2차원으로 반환됨 [[nan, nan, ..., nan]]
        
        for data_idx, label in enumerate(labels):
            self.data_matrix[label, np.where(np.isnan(self.data_matrix[label]))[0][0]] = data_idx 
            self.num_per_class[label] += 1
        
        self.valid_classes = [c.item() for c, count in zip(self.classes, self.num_per_class) if count >= self.num_shots+self.num_queries]
        
    def __iter__(self):
        for _ in range(self.episodes):
            tasks = []
            for t in range(self.num_tasks):
                batch_support_set = torch.LongTensor(self.num_ways*self.num_shots)
                batch_query_set = torch.LongTensor(self.num_ways*self.num_queries)
                
                way_indices = torch.randperm(len(self.valid_classes))[:self.num_ways]
                selected_classes = [self.valid_classes[idx] for idx in way_indices]
                
                for i, label in enumerate(selected_classes):
                    slice_for_support = slice(i*self.num_shots, (i+1)*self.num_shots)
                    slice_for_queries = slice(i*self.num_queries, (i+1)*self.num_queries)
                    
                    samples = torch.randperm(self.num_per_class[label])[:self.num_shots+self.num_queries]
                    batch_support_set[slice_for_support] = self.data_matrix[label][samples][:self.num_shots]
                    batch_query_set[slice_for_queries] = self.data_matrix[label][samples][self.num_shots:]
                
                batch = torch.cat((batch_support_set, batch_query_set))
                tasks.append(batch)
            
            batches = torch.cat(tasks)
            yield batches
            
    def __len__(self):
        return self.episodes

# 같은 class에서 서로 다른 sample 2개를 뽑기
class ClassSampler(Sampler):
    def __init__(self, labels, episodes, batch_size, data_source=None):
        super().__init__(data_source)
        self.labels = labels
        self.episodes = episodes
        self.batch_size = batch_size
        
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.num_classes = len(self.classes)
        self.data_matrix = torch.Tensor(np.empty((len(self.classes), max(self.counts)), dtype=int)*np.nan)
        self.num_per_class = torch.zeros_like(self.classes)
        
        for data_idx, label in enumerate(labels):
            self.data_matrix[label, np.where(np.isnan(self.data_matrix[label]))[0][0]] = data_idx 
            self.num_per_class[label] += 1
        
        self.valid_classes = [c.item() for c, count in zip(self.classes, self.num_per_class) if count >= 2]
        
    def __iter__(self):
        for i in range(self.episodes):
            x_batch = torch.LongTensor(self.batch_size)
            y_batch = torch.LongTensor(self.batch_size)
            for j in range(self.batch_size):
                idx = torch.randperm(len(self.valid_classes))[0]
                selected_class = self.valid_classes[idx]
                
                samples_idx = torch.randperm(self.num_per_class[selected_class])[:2]
                x_batch[j] = self.data_matrix[selected_class][samples_idx][0]
                y_batch[j] = self.data_matrix[selected_class][samples_idx][1]
            
            yield torch.cat((x_batch, y_batch))
            
    def __len__(self):
        return self.batch_size


def load_dataset(args, dataset):
    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    if dataset == 'cifar10':
        ssltransform = SSLTransform(32, args.model)
        trainset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=True, download=True, transform=ssltransform)
        testset = torchvision.datasets.CIFAR10(root = '../data/cifar10', train=False, download=True, transform=transform_test)
        
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
        valloader = None
        
        num_classes = 10
    
    elif dataset == 'miniimagenet':
        if args.model == 'ijepa':
            trainloader, testloader, valloader, num_classes = load_ijepa_data(args)
        elif args.model == 'setfsl':
            trainloader, testloader, valloader, num_classes = load_setfsl_data(args)
        else:
            ssltransform = SSLTransform(args.img_size, args.model)
            
            transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            transform_test = transforms.Compose([
                transforms.Resize(args.img_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
            
            if args.model == 'mae':
                transform_train = transform_train
            else:
                transform_train = ssltransform
            
            trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_train)
            testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
            valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
            num_classes = 64
            
            testset_labels = torch.LongTensor(testset.targets)
            valset_labels = torch.LongTensor(valset.targets)
            
            test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 600, num_tasks=1)
            val_sampler = FewShotSampler(valset_labels, args.test_num_ways, args.num_shots, args.num_queries, 100, num_tasks=1)
            
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
            valloader = DataLoader(valset, batch_sampler=val_sampler, pin_memory=True)
            
            if args.test == 'fewshot' or args.test == 'crossdomain':
                testloader = DataLoader(testset, batch_sampler=test_sampler, pin_memory=True)
            else:
                testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
        
    else:
        trainloader = None
        valloader = None
        num_classes = None
        
        transform_test = transforms.Compose([ 
            transforms.Resize(args.img_size, antialias=True),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        
        testset = cd_dataset.load_crossdomain_dataset(dataset, transform_test)
            
        testset_labels = torch.LongTensor(testset.targets)
        test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 600, num_tasks=1)
        testloader = DataLoader(testset, batch_sampler=test_sampler, pin_memory=True)
    
    return trainloader, testloader, valloader, num_classes


def load_ijepa_data(args):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.3, 1.0), interpolation=3),  # 3 is bicubic
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform_test = transforms.Compose([
        transforms.Resize(args.img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
    valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
    num_classes = 64
    
    testset_labels = torch.LongTensor(testset.targets)
    valset_labels = torch.LongTensor(valset.targets)
    
    test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 600, num_tasks=1)
    val_sampler = FewShotSampler(valset_labels, args.test_num_ways, args.num_shots, args.num_queries, 100, num_tasks=1)
    
    mask_collator = MBMaskCollator(input_size=(args.img_size, args.img_size), patch_size=args.patch_size, allow_overlap=False)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn = mask_collator, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    valloader = DataLoader(valset, batch_sampler=val_sampler, pin_memory=True)
    
    if args.test == 'fewshot' or args.test == 'crossdomain':
        testloader = DataLoader(testset, batch_sampler=test_sampler, pin_memory=True)
    else:
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
    
    return trainloader, testloader, valloader, num_classes


def load_setfsl_data(args):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(0.3, 1.0), interpolation=3),  # 3 is bicubic
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform_test = transforms.Compose([
        transforms.Resize(args.img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    
    trainset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/test', transform=transform_test)
    valset = torchvision.datasets.ImageFolder(root='../data/fewshotdata/miniimagenet/data/val', transform=transform_test)
    num_classes = 64
    
    trainset_labels = torch.LongTensor(trainset.targets)
    testset_labels = torch.LongTensor(testset.targets)
    valset_labels = torch.LongTensor(valset.targets)
    
    train_sampler = ClassSampler(trainset_labels, 75, args.batch_size)
    test_sampler = FewShotSampler(testset_labels, args.test_num_ways, args.num_shots, args.num_queries, 600, num_tasks=1)
    val_sampler = FewShotSampler(valset_labels, args.test_num_ways, args.num_shots, args.num_queries, 100, num_tasks=1)
    
    trainloader = DataLoader(trainset, batch_sampler=train_sampler, num_workers=10, pin_memory=True)
    valloader = DataLoader(valset, batch_sampler=val_sampler, num_workers=10, pin_memory=True)
    
    if args.test == 'fewshot' or args.test == 'crossdomain':
        testloader = DataLoader(testset, batch_sampler=test_sampler, num_workers=10, pin_memory=True)
    else:
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=10)
    
    return trainloader, testloader, valloader, num_classes
