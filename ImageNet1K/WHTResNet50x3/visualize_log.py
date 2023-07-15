# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:52:01 2022

@author: HP
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', metavar='ARCH', default='dct_resnet18')
    parser.add_argument('--save_dir', type=str, default='./saved')
    args = parser.parse_args()
    model = args.arch
    log_dir = args.save_dir
    
    log_train_loss = np.load(os.path.join(log_dir, model+'_log_train_loss.npy'))
    log_train_acc1 = np.load(os.path.join(log_dir, model+'_log_train_acc1.npy'))
    log_train_acc5 = np.load(os.path.join(log_dir, model+'_log_train_acc5.npy'))
    log_val_loss = np.load(os.path.join(log_dir, model+'_log_val_loss.npy'))
    log_val_acc1 = np.load(os.path.join(log_dir, model+'_log_val_acc1.npy'))
    log_val_acc5 = np.load(os.path.join(log_dir, model+'_log_val_acc5.npy'))
    
    n_epochs = len(log_train_acc1)
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.plot(np.arange(1, n_epochs + 1), log_train_loss)  # train loss (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), log_val_loss)         #  test loss (on epoch end)
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Valid'], loc="upper left")
    
    plt.subplot(132)
    plt.plot(np.arange(1, n_epochs + 1), 100-log_train_acc1)  # train loss (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), 100-log_val_acc1)         #  test loss (on epoch end)
    plt.title("Center-Crop Top-1 Error")
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Valid'], loc="upper left")
    
    plt.subplot(133)
    plt.plot(np.arange(1, n_epochs + 1), 100-log_train_acc5)  # train loss (on epoch end)
    plt.plot(np.arange(1, n_epochs + 1), 100-log_val_acc5)         #  test loss (on epoch end)
    plt.title("Center-Crop Top-5 Error")
    plt.xlabel('Epoch')
    plt.ylabel('Error (%)')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Valid'], loc="upper left")