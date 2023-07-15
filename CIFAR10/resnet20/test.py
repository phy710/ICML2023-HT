#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:24:45 2022

@author: Zephyr
"""
import os
import torch
from tqdm import tqdm
from torch import nn

from resnet import ResNet20
from read_data import read_cifar10

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x,labels = sample
            x,labels = x.to(device), labels.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, labels)
            _,pred = torch.max(outputs,1)
            num_correct = (pred == labels).sum()
            
            loss = loss.item()
            acc = num_correct.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc += num_correct.item()
            pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
    
    return test_loss/count, test_acc/count

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    save_dir = './saved'
    data_dir = '../data'
    batchsize = 128

    
    _, test_generator = read_cifar10(batchsize,data_dir)
    
    model = ResNet20().to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'acc_model.pth')))
    print(model)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    test_loss, test_acc = test_loop(test_generator, model, criterion, device)
    print("\nTest loss: {:f}, acc: {:f}".format(test_loss, test_acc))   