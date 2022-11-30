import torch
import numpy as np
import pandas as pd
import random
# Custom data from csv file
class EstrusDataset():
    def __init__(self,path, mode):
        self.path = path
        self.mode = mode

        self.dataset, self.label, self.target = self.read_csv(self.path)
        
    def check_data(self,index,hours):     # use to check data is complete 
        for i in range(1,25,1):
            #print(hours)
            if(int(hours[index:index+1].values.tolist()[0][0]) != i):
                return 0
            index = index + 1
        return 1
        
    def read_csv(self,data_path):
        
        # loader
        training = []
        training_one = []
        label = []
        train_dataset = []         # last type of data
        estrus = []
        target = []
        
        trainnor = 0
        trainnor_lst = []
        trainEstrus = 0
        trainEstrusr_lst = []
        #prepare training data

        self.hours       = pd.read_csv(data_path, usecols=["hour"])         # use to check data is continuous or not
        self.in_alleys   = pd.read_csv(data_path, usecols=["IN_ALLEYS"])    # train dataset
        self.rest        = pd.read_csv(data_path, usecols=["REST"])         # train dataset
        self.eat         = pd.read_csv(data_path, usecols=["EAT"])  # train dataset
        self.oestrusl    = pd.read_csv(data_path, usecols=["oestrus"])      # labels
        


        if(self.mode == "train"):
            while(trainnor<204):      
                index = random.randint(0,len(self.hours)-200000)
                if (index not in trainnor_lst) and (self.check_data(index,self.hours)==1) and ((self.oestrusl[index:index+1].values.tolist()[0][0])==0):
                    save = 1
                    trainnor += 1
                    trainnor_lst.append(index)
                else:
                    save = 0
        
                if(save):    
                    target.append(0)
                    estrus.append(1)
                    estrus.append(0)
                    label.append(estrus)   # ground truth
                    estrus = []
                    
    
                    for j in range(index,index+24,1):
                        in_alleys_v = self.in_alleys[j:j+1].values.tolist()[0][0]
                        rest_v = self.rest[j:j+1].values.tolist()[0][0]
                        eat_v = self.eat[j:j+1].values.tolist()[0][0]

                        training.append(in_alleys_v)
                        training.append(rest_v)
                        training.append(eat_v)
                        #training.append(activity_v)
                    train_dataset.append(training)
                    training = []


            index = 0 
            while(trainEstrus<204):      
                
                if (self.check_data(index,self.hours)==1) and ((self.oestrusl[index:index+1].values.tolist()[0][0])==1):
                    save = 1
                    trainEstrus += 1
                    trainEstrusr_lst.append(index)
                else:
                    save = 0
        
                if(save):    
                    target.append(1)
                    estrus.append(0)
                    estrus.append(1)
                    label.append(estrus)   # ground truth
                    estrus = []
                    
                    for j in range(index,index+24,1):
                        in_alleys_v = self.in_alleys[j:j+1].values.tolist()[0][0]
                        rest_v = self.rest[j:j+1].values.tolist()[0][0]
                        eat_v = self.eat[j:j+1].values.tolist()[0][0]

                        training.append(in_alleys_v)
                        training.append(rest_v)
                        training.append(eat_v)
                        #training.append(activity_v)
                    train_dataset.append(training)
                    training = []
                index += 1

            train_dataset = torch.Tensor(train_dataset)
            label = torch.Tensor(label)
            print(f"All {self.mode} data are prepared. length is {len(train_dataset)}")


        if(self.mode == "test"):
            while(trainnor<53):      
                index = random.randint(len(self.hours)-200000,len(self.hours)-24)
                if (index not in trainnor_lst) and (self.check_data(index,self.hours)==1) and ((self.oestrusl[index:index+1].values.tolist()[0][0])==0):
                    save = 1
                    trainnor += 1
                    trainnor_lst.append(index)
                else:
                    save = 0
        
                if(save):    
                    target.append(0)
                    estrus.append(1)
                    estrus.append(0)
                    label.append(estrus)   # ground truth
                    estrus = []
                    
    
                    for j in range(index,index+24,1):
                        in_alleys_v = self.in_alleys[j:j+1].values.tolist()[0][0]
                        rest_v = self.rest[j:j+1].values.tolist()[0][0]
                        eat_v = self.eat[j:j+1].values.tolist()[0][0]

                        training.append(in_alleys_v)
                        training.append(rest_v)
                        training.append(eat_v)
                        #training.append(activity_v)
                    train_dataset.append(training)
                    training = []


            index = 0 
            while(trainEstrus<257):  # all estrus data in dataset      
                
                if (self.check_data(index,self.hours)==1) and ((self.oestrusl[index:index+1].values.tolist()[0][0])==1):
                    save = 1
                    trainEstrus += 1
                    trainEstrusr_lst.append(index)
                else:
                    save = 0
        
                if(save and trainEstrus>204):    
                    target.append(1)
                    estrus.append(0)
                    estrus.append(1)
                    label.append(estrus)   # ground truth
                    estrus = []
                    
                    for j in range(index,index+24,1):
                        in_alleys_v = self.in_alleys[j:j+1].values.tolist()[0][0]
                        rest_v = self.rest[j:j+1].values.tolist()[0][0]
                        eat_v = self.eat[j:j+1].values.tolist()[0][0]

                        training.append(in_alleys_v)
                        training.append(rest_v)
                        training.append(eat_v)
                        #training.append(activity_v)
                    train_dataset.append(training)
                    training = []

                index += 1

            train_dataset = torch.Tensor(train_dataset)
            label = torch.Tensor(label)
            print(f"All {self.mode} data are prepared. length is {len(train_dataset)}")
        
        return train_dataset, label , target
        
        
    def __getitem__(self, index):
        # print("index now = ",index)
        # print("self.label[index]= ",self.label[index])
        return self.dataset[index]/1000, self.label[index], self.target[index]
        
        
    def __len__(self):
        return len(self.dataset)