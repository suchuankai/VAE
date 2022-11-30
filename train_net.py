import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

#------others-----
import net 
import dataload
import dataload_pool
import dataload_pool_reshpae



class Train_VAE():     # trainimg progress will at this program and called by start.py

    def __init__(self,opt):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = opt                          # get hyperparameters from start.py
        self.train_log = "\nepochs: " + str(self.opt.n_epochs) + "\nbatch size: " + str(self.opt.batch_size) \
                         + "\ntraining dataset: " + self.opt.train_path + "\ntesting dataset: " + self.opt.test_path
        # self.model = net.VAE(image_size=3, h_dim=400, z_dim=20)  
        self.model = net.VAE(image_size=72, h_dim=400, z_dim=20)   
        # print(self.model)
        self.model = self.model.to(self.DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = 0.0005)
        self.criterion = nn.BCELoss()
        
        self.train_model()
        

    def load_training_data(self, path):
        dataset = dataload_pool_reshpae.EstrusDataset(path, mode="train")        
        self.train_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, drop_last=True)
        

    def load_testing_data(self, path):
        dataset = dataload_pool_reshpae.EstrusDataset(path, mode="test")
        self.test_loader = DataLoader(dataset=dataset, batch_size=self.opt.batch_size, drop_last=True)
    

    def test_one_epoch(self, model, data_loader, device, state):

        device = self.DEVICE
        correct_pred, num_examples = 0, 0
        loss_sum = 0.
        accuracy = 0

        if(state=='train'):
            model.train()
        else:
            model.eval()

        for batch_index, (data, label, target) in enumerate(data_loader):

            data = data.to(self.DEVICE)
            label = label.to(self.DEVICE)
            target = target.to(self.DEVICE)
            logits = self.model(data)           # Data has not yet passed sigmoid
            

            
            
            cost = self.criterion(logits, label)
            loss_sum += cost.item()
            _, predicted_labels = torch.max(logits,1)
            
            correct_pred += (predicted_labels == target).sum()
            #print(target,target.size())
            #print("presdict: ",predicted_labels,predicted_labels.size())
            
            num_examples += label.size(0)
            if(state == 'train'):
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()
            
 
        return correct_pred / num_examples *100, loss_sum / num_examples


   
    
    def train_model(self):
        

        train_acc_lst, valid_acc_lst   = [], []
        train_loss_lst, valid_loss_lst = [], []
        
        print("Start training...")

        

        for epoch in range(self.opt.n_epochs):
            self.load_training_data(self.opt.train_path)
            self.load_testing_data(self.opt.test_path)
            
            # Train
            self.model.train()
            train_acc, train_loss = self.test_one_epoch(self.model, self.train_loader, device = self.DEVICE, state='train')

            train_acc_lst.append(train_acc.cpu().item())
            train_loss_lst.append(train_loss)
            
            # Test
            self.model.eval()
            with torch.set_grad_enabled(False):
                valid_acc, valid_loss = self.test_one_epoch(self.model, self.test_loader, device = self.DEVICE, state= 'test')

                valid_acc_lst.append(valid_acc.cpu().item())  
                valid_loss_lst.append(valid_loss) 
                
            if(epoch % 5  ==0):
                print(f"Epoch: {epoch} train_acc: {train_acc} | train_loss: {train_loss} | valid_acc: {valid_acc} | valid_loss:{valid_loss}")
                
        
        path = self.save_model()
        if(self.opt.plot):
            self.plot(train_acc_lst, valid_acc_lst, train_loss_lst, valid_loss_lst, path)
        print("complete training.")

    def takeSecond(elem):
        return elem[[-1][6:]]


    def save_model(self):
        path = "./result/"
        count = os.listdir(path)
        
        if(len(count)==0):
            model_path = path + "train_1"
        else:
            max = 0
            for i in range(len(count)):
                if(len(count[i]) < 7):   # smaller than 10
                    if int(count[i][6]) > max:
                        max = int(count[i][6])
                else:                    # bigger than 10
                    if int(count[i][6:]) > max:
                        max = int(count[i][6:])
            index = int(max) + 1         # get the last index
            model_path = path + "train_" + str(index)

        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        with open(model_path + './time.txt', 'a+') as f:
            seconds = time.time()
            local_time = time.ctime(seconds)
            f.write("本地時間：" + str(local_time))
            f.write(self.train_log)

        torch.save(self.model.state_dict(),model_path + "./model.pt")
        return model_path


    def plot(self,train_acc_lct, valid_acc_lct, train_loss_lct, valid_loss_lct, path):
        # print(path)
        # print(type(path))
        
        plt.plot(range(1,self.opt.n_epochs+1),train_acc_lct, label="Training accuracy")
        plt.plot(range(1,self.opt.n_epochs+1),valid_acc_lct, label="Validation accuracy")
        plt.legend(loc='lower right')
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')
        plt.savefig(path + "/acc.png")
        plt.show()

        plt.plot(range(1,self.opt.n_epochs+1),train_loss_lct, label="Training loss")
        plt.plot(range(1,self.opt.n_epochs+1),valid_loss_lct, label="Validation loss")
        plt.legend(loc='upper right')
        plt.ylabel('loss')
        plt.xlabel('Epoch')
        plt.savefig(path +"/loss.png")
        plt.show()
