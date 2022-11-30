# Test dataloader and use to test trained model

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import dataloader_test_triple
import dataload_test
import dataload
import numpy

from others.PR_curve import PR_curve
import torch
import net
 

model_path = "./result/train_4/model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.VAE(image_size=3, h_dim=400, z_dim=20)
model = model.to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()


dataset = dataload.EstrusDataset("./data/dataset4-1.csv",mode= "test")
train_loader = DataLoader(dataset=dataset, batch_size=1)

# dataset = dataload.EstrusDataset("./data/dataset3-1.csv",mode= "test")
# test_loader = DataLoader(dataset=dataset, batch_size=1)

normal_total = 0
estrus_total = 0
estrus = [0] * 12
estrus_avg = 0
normal = [0] * 12
normal_avg = 0

for epoch in range(1):
  
    for step, (data, label, target) in enumerate(train_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        soft_out = model(data)
        soft_out = torch.squeeze(soft_out)

       
        if(target):
            estrus_total+= 1
            estrus_avg += soft_out[1].cpu().item()
            if(soft_out[1].cpu().item()<0.1):
                estrus[0]+= 1
            elif(soft_out[1].cpu().item()<0.2):
                estrus[1]+= 1
            elif(soft_out[1].cpu().item()<0.3):
                estrus[2]+= 1
            elif(soft_out[1].cpu().item()<0.4):
                estrus[3]+= 1
            elif(soft_out[1].cpu().item()<0.5):
                estrus[4]+= 1
            elif(soft_out[1].cpu().item()<0.6):
                estrus[5]+= 1
            elif(soft_out[1].cpu().item()<0.7):
                estrus[6]+= 1
            elif(soft_out[1].cpu().item()<0.8):
                estrus[7]+= 1
            elif(soft_out[1].cpu().item()<0.9):
                estrus[8]+= 1
            elif(soft_out[1].cpu().item()<1):
                estrus[9]+= 1

        else:
            normal_total+=1
            normal_avg += soft_out[1].cpu().item()
            if(soft_out[1].cpu().item()<0.1):
                normal[0]+= 1
            elif(soft_out[1].cpu().item()<0.2):
                normal[1]+= 1
            elif(soft_out[1].cpu().item()<0.3):
                normal[2]+= 1
            elif(soft_out[1].cpu().item()<0.4):
                normal[3]+= 1
            elif(soft_out[1].cpu().item()<0.5):
                normal[4]+= 1
            elif(soft_out[1].cpu().item()<0.6):
                normal[5]+= 1
            elif(soft_out[1].cpu().item()<0.7):
                normal[6]+= 1
            elif(soft_out[1].cpu().item()<0.8):
                normal[7]+= 1
            elif(soft_out[1].cpu().item()<0.9):
                normal[8]+= 1
            elif(soft_out[1].cpu().item()<1):
                normal[9]+= 1

path = './test/train_model_PR_5.txt'

with open(path, 'a+') as f:
    
    f.write("Estrus data start:\n")
    for i in range(10):
        print(f"estrus{i} is {estrus[i]}.")
        percent = str(i*10) + '~'  + str((i+1)*10) + '%: ' 
        f.write(percent)
        f.write(str(estrus[i]))
        f.write("\n")
    print("estrus total = ",estrus_total)
    print("estrus_avg = ",estrus_avg / estrus_total)
    f.write("\nestrus total =  ")
    f.write(str(estrus_total))
    f.write("\n")
    f.write("estrus_avg =  ")
    f.write(str(estrus_avg / estrus_total))


    f.write("\n\nNormal data start:\n")
    for i in range(10):
        print(f"normal{i} is {normal[i]}.")
        percent = str(i*10) + '~'  + str((i+1)*10) + '%: ' 
        f.write(percent)
        f.write(str(normal[i]))
        f.write("\n")
    print("normal total = ",normal_total)
    f.write("\nnormal_total = ")
    f.write(str(normal_total))
    print("normal_avg = ",normal_avg / normal_total)
    f.write("\nnormal_avg = ")
    f.write(str(normal_avg / normal_total))

print("done")
pr = PR_curve(path=path)
