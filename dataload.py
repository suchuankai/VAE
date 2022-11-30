import torch
import numpy as np
import pandas as pd

# Custom data from csv file
class EstrusDataset():
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.dataset, self.label, self.target = self.read_csv(self.path)
        
    def check_data(self,index,hours):     # use to check data is complete 
        for i in range(1,25,1):
            if(int(hours[index:index+1].values.tolist()[0][0]) != i):
                return 0
            index = index + 1
        return 1
        
    def read_csv(self,data_path):
        
        # debug parameters
        line = 0                   # use to check data index directly
        count = [0] * 25           # check each hours are same or not
        oestrus_count = 0          # count oestrus 
        trainEstrus   = 0
        
        # loader
        training = []
        training_one = []
        label = []
        train_dataset = []         # last type of data
        estrus = []
        target = []
        
        hours       = pd.read_csv(data_path, usecols=["hour"])         # use to check data is continuous or not
        in_alleys   = pd.read_csv(data_path, usecols=["IN_ALLEYS"])    # train dataset
        rest        = pd.read_csv(data_path, usecols=["REST"])         # train dataset
        eat         = pd.read_csv(data_path, usecols=["EAT"])  # train dataset
        oestrusl    = pd.read_csv(data_path, usecols=["oestrus"])      # labels
        
        for i in range(len(hours)):  
            # save = self.check_data(i,hours)       

            if(self.mode == "train"):
                if i<= len(hours)-200000 and self.check_data(i,hours)==1:
                    save = 1
                    trainEstrus = trainEstrus+1
                else:
                    save = 0
            else:
                if i > len(hours)-200000 and self.check_data(i,hours)==1:
                    save = 1
                    trainEstrus = trainEstrus+1
                else:
                    save = 0
           
            if(save):    
                line += 1
                target.append(int(oestrusl[i:i+1].values.tolist()[0][0]))
                if(int(oestrusl[i:i+1].values.tolist()[0][0])):
                    estrus.append(0)
                    estrus.append(1)
                else:
                    estrus.append(1)
                    estrus.append(0)
                label.append(estrus)   # ground truth
                estrus = []
                
                if(int(oestrusl[i:i+1].values.tolist()[0][0])):            # check estrus line
                    oestrus_count += 1
                    #print(f"now is line {line}.")
                    
                for j in range(i,i+24,1):
                    count[int(hours[j:j+1].values.tolist()[0][0])] += 1
                    in_alleys_v = in_alleys[j:j+1].values.tolist()[0][0]
                    rest_v = rest[j:j+1].values.tolist()[0][0]
                    eat_v = eat[j:j+1].values.tolist()[0][0]
                    training.append(in_alleys_v)
                    training.append(rest_v)
                    training.append(eat_v)
                    training_one.append(training)
                    training = []
                train_dataset.append(training_one)
                training_one = []
                
        train_dataset = torch.Tensor(train_dataset)
        label = torch.Tensor(label)
        print(f"All {self.mode} data are prepared.")
        
        return train_dataset, label , target
        
        
    def __getitem__(self, index):
        return self.dataset[index]/1000, self.label[index], self.target[index]
        
        
    def __len__(self):
        return len(self.dataset)