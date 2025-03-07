import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import torch.nn.utils as utils
import torch.optim as optim
from tqdm import tqdm

import dataloader

from utils import parsing_argument, load_model, set_parameters, split_support_query_set

from torch.utils.tensorboard import SummaryWriter


def train_per_epoch(args, dataloader, net, optimizer, scheduler, device):
    running_loss = 0.0
    net.train()
    
    representations = None
    label_list = None
    is_tuple = False
    
    for data in dataloader:
        inputs, labels = data
        
        if args.model != 'ijepa':
            if isinstance(inputs, list):
                is_tuple = True
                inputs = torch.stack(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
        
        if args.model == 'setfsl':
            loss = net(inputs, labels, device)
        else:
            loss = net(inputs, device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.test == 'knn':
            if is_tuple:
                inputs = inputs[2]
            with torch.no_grad():
                if representations == None:
                    representations = net.encoding(inputs)
                    label_list = labels
                else:
                    representations = torch.cat((representations, net.encoding(inputs)), dim=0)
                    label_list = torch.cat((label_list, labels), dim=0)
        
        running_loss += loss.item()
        
    return running_loss/len(dataloader), representations, label_list

def knn_test(testloader, net, representations, label_list, device):
    total = 0
    correct = 0
    net.eval()
    
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            x = net.encoding(inputs)
            distances = torch.cdist(x, representations)
            min_indices = torch.argmin(distances, dim=-1)
            predictions = torch.tensor([label_list[idx].item() for idx in min_indices]).to(device)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    acc = 100 * correct / total
    
    return acc


def fewshot_test(testloader, net, args, device):
    total_acc = 0
    
    if args.adaptation:
        accuracy = net.ft_fewshot_acc(testloader, device, n_iters=100, args=args)
    else:
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            net.eval()
            if args.model == 'protonet':
                with torch.no_grad():
                    acc = net(args, inputs, labels, args.test_num_ways, device)
            else:
                acc = net.fewshot_acc(args, inputs, labels, device)
            total_acc += acc
        accuracy = total_acc/len(testloader)
    
    return accuracy

def crossdomain_test(args, net, device, outputs_log):
    print('--- crossdomain test ---')
    if args.dataset == 'BSCD':
        dataset_list = ['CropDisease','EuroSAT', 'ISIC', 'ChestX', 'miniimagenet']
    elif args.dataset == 'FWT':
        dataset_list = ['cub', 'cars', 'places', 'plantae']
    elif args.dataset == 'all':
        dataset_list = ['CropDisease','EuroSAT', 'ISIC', 'ChestX','cub', 'cars', 'places', 'plantae']
    else:
        print('invalid dataset')
        return
    
    total = 0
    for dataset in dataset_list:
        trainloader, testloader, valloader, num_classes = dataloader.load_dataset(args, dataset)
        print(f'--- {dataset} test ---')
        acc = fewshot_test(testloader, net, args, device=device)
        total += acc
        print(f'{dataset} fewshot_acc : %.3f'%(acc))
        print(f'{dataset} fewshot_acc : %.3f'%(acc), file=outputs_log)
    mean = total / len(dataset_list)
    print(f'mean_acc : %.3f'%(mean))
    print(f'mean_acc : %.3f'%(mean), file=outputs_log)


def train(args, trainloader, testloader, valloader, net, optimizer, scheduler, device, writer, outputs_log):
    max_acc = 0
    for epoch in range(args.epochs):
        running_loss, representations, label_list = train_per_epoch(args, trainloader, net, optimizer, scheduler, device)
        
        acc = 0
        if args.test == 'knn':
            acc = knn_test(testloader, net, representations, label_list, device)
            writer.add_scalar('train / KNN_acc', acc, epoch+1)
        elif args.test == 'fewshot' and (epoch+1) % 5 == 0 or epoch == 0:
            #print(F.softmax(net.decoder.weight_for_blk))
            acc = fewshot_test(valloader, net, args, device=device)
            writer.add_scalar('train / fewshot_acc', acc, epoch+1)
        
        if scheduler:
            scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print('epoch[%d] - training loss : %.3f / %s_acc : %.3f / lr : %f'%(epoch+1, running_loss, args.test, acc, lr))
        print('epoch[%d] - training loss : %.3f / %s_acc : %.3f'%(epoch+1, running_loss, args.test, acc), file=outputs_log)
        writer.add_scalar('train / train_loss', running_loss, epoch+1)
        writer.add_scalar('train / learning_rate', lr, epoch+1)
        
        torch.save(net.state_dict(), f'./{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.log}.pt')
        '''
        if acc > max_acc:
            torch.save(net.state_dict(), f'./{args.model}_best_ep_{args.learningrate}lr_{args.log}.pt')
        '''
        running_loss = 0.0
        
    print('Training finished',file=outputs_log)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if args.train:
        outputs_log = open(f'./outputs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.log}_{cur_time}.txt','w')
        writer = SummaryWriter(f'./logs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.log}_{cur_time}')
    elif args.test:
        outputs_log = open(f'./outputs/{args.model}_test_{args.dataset}_{cur_time}_{args.log}.txt','w')
        writer = None
    
    net = load_model(args)
    net.to(device)
    
    trainable_parameters = sum(p.numel() for p in net.encoder.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in net.encoder.parameters())
    print(trainable_parameters, total_parameters)
    #net.load_state_dict(torch.load('setfsl_300ep_0.001lr_test_layer11_mean_withnorm.pt'), strict= False)
    
    net.encoder.load_state_dict(torch.load('./checkpoint/dino_deitsmall16_pretrain.pth'), strict= False)
    #net.target_encoder.load_state_dict(torch.load('./checkpoint/dino_deitsmall16_pretrain.pth'))
    
    if args.train:
        trainloader, testloader, valloader, num_classes = dataloader.load_dataset(args, args.dataset)
        optimizer,scheduler = set_parameters(args, net, len(trainloader))
        if hasattr(net, 'ipe'):
            net.ipe = len(trainloader)
        print(f"--- train ---")
        #data loader 600으로 변경하기
        #_, valloader, _, _ = dataloader.load_dataset(args, 'CropDisease')
        train(args, trainloader, testloader, valloader, net, optimizer, scheduler, device, writer, outputs_log)
    
    if args.test == 'fewshot':
        trainloader, testloader, valloader, num_classes = dataloader.load_dataset(args, args.dataset)
        print(f'--- {args.dataset} test ---')
        acc = fewshot_test(testloader, net, args, device)
        print('fewshot_acc : %.3f'%(acc))
        print('fewshot_acc : %.3f'%(acc), file=outputs_log)
    
    elif args.test == 'crossdomain':
        crossdomain_test(args, net, device, outputs_log)
        
    outputs_log.close()
    if writer != None:
        writer.close()

if __name__ == "__main__":
    main()
