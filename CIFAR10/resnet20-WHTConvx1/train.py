#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:35:23 2022

@author: innopeak
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import numpy as np
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from torch import nn

from resnet_WHT import ResNet20
from read_data import read_cifar10

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    for batch, sample in enumerate(pbar):
        x,labels = sample
        x,labels = x.to(device), labels.to(device)

        outputs = model(x)
        loss = loss_fn(outputs, labels)
        _,pred = torch.max(outputs,1)
        num_correct = (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        acc = num_correct.item()/len(labels)
        count += len(labels)
        train_loss += loss*len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
    return train_loss/count, train_acc/count
        
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
    data_dir = '../../data'
    batchsize = 128
    n_epochs = 200
    Lr = 0.1
    evaluate_train = False
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    train_generator,test_generator = read_cifar10(batchsize,data_dir)
    
    model = ResNet20().to(device)
    print(model)
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    if torch.cuda.is_available() and torch.cuda.device_count()>1:
      print("Using {} GPUs.".format(torch.cuda.device_count()))
      model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss().to(device)   
    weight_p, bias_p = [],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p +=[p]
        else:
            weight_p +=[p]
    optimizer = torch.optim.SGD([{'params':weight_p,'weight_decay':1e-4},
                                 {'params':bias_p,'weight_decay':0}],lr=Lr,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[82,122,163],gamma=0.1,last_epoch=-1)

    idx_best_loss = 0
    idx_best_acc = 0
    
    log_train_loss = []
    log_train_acc = []
    log_test_loss = []
    log_test_acc = []
    
    for epoch in range(1, n_epochs+1):
        print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, scheduler.get_last_lr()))
        train_loss, train_acc = train_loop(train_generator, model, criterion, optimizer, device)
        if evaluate_train:
            train_loss, train_acc = test_loop(train_generator, model, criterion, device)
        test_loss, test_acc = test_loop(test_generator, model, criterion, device)
        print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        print("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))
        scheduler.step()
        
        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_test_loss.append(test_loss)
        log_test_acc.append(test_acc)
        
        if test_loss <= log_test_loss[idx_best_loss]:
            print("Save loss-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'loss_model.pth'))
            idx_best_loss = epoch - 1
        
        if test_acc >= log_test_acc[idx_best_acc]:
            print("Save acc-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'acc_model.pth'))
            idx_best_acc = epoch - 1
        print("")

    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(log_train_loss[idx_best_loss], log_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc: {:f}".format(log_test_loss[idx_best_loss], log_test_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(log_train_loss[idx_best_acc], log_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc: {:f}".format(log_test_loss[idx_best_acc], log_test_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(log_train_loss[-1], log_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc: {:f}".format(log_test_loss[-1], log_test_acc[-1]))           
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    log_train_loss = np.array(log_train_loss)
    log_train_acc = np.array(log_train_acc)
    log_test_loss = np.array(log_test_loss)
    log_test_acc = np.array(log_test_acc)
    np.save(os.path.join(save_dir, 'log_train_loss.npy'), log_train_loss)
    np.save(os.path.join(save_dir, 'log_train_acc.npy'), log_train_acc)
    np.save(os.path.join(save_dir, 'log_test_loss.npy'), log_test_loss)
    np.save(os.path.join(save_dir, 'log_test_acc.npy'), log_test_acc)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.arange(1, n_epochs + 1), log_train_loss)  # train loss (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_test_loss)         #  test loss (on epoch end)
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Test'], loc="upper left")
    
    plt.subplot(122)

    plt.plot(np.arange(1, n_epochs + 1), log_train_acc)  # train accuracy (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_test_acc)         #  test accuracy (on epoch end)
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Test'], loc="upper left")            
    
    plt.savefig(os.path.join(save_dir, 'log.png'))
    plt.show()
