A] Diamond Architecture[Best Architecture(85.8%)]:

context = 32
 
{
   nn.Linear(in_size,512),
   nn.BatchNorm1d(512,affine=False)
   nn.LeakyReLU(),
   nn.Dropout(p=0.1), 

   nn.Linear(512,1024), 
   nn.BatchNorm1d(1024,affine=False), 
   nn.LeakyReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(1024,2048), 
   nn.BatchNorm1d(2048,affine=False), 
   nn.ReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(2048,4096), 
   nn.BatchNorm1d(4096,affine=False),
   nn.ReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(4096,2048), 
   nn.BatchNorm1d(2048,affine=False), 
   nn.ReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(2048,1024), 
   nn.BatchNorm1d(1024,affine=False), 
   nn.ReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(1024,512), 
   nn.BatchNorm1d(512,affine=False), 
   nn.LeakyReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(512,256), 
   nn.BatchNorm1d(256,affine=False), 
   nn.LeakyReLU(),
   nn.Dropout(p=0.1),

   nn.Linear(256,40),
}

B] How to Run the Code:

Part 1:

!mkdir data

import json

TOKEN = {"username":"mangalamsahai","key":"521f66540469b3a12f7b11566d8b1c14"}

! pip install kaggle==1.5.12
! mkdir -p .kaggle
! mkdir -p /content & mkdir -p /content/.kaggle & mkdir -p /root/.kaggle

with open('/content/.kaggle/kaggle.json','w') as file:
    json.dump(TOKEN, file)

! pip install --upgrade --force-reinstall --no-deps kaggle
! ls "/content/.kaggle"
! chmod 600 /content/.kaggle/kaggle.json
! cp /content/.kaggle/kaggle.json /root/.kaggle/

! kaggle config set -n path -v /content

! unzip competitions/11-785-s22-hw1p2/11-785-s22-hw1p2.zip

! kaggle competitions download -c 11-785-s22-hw1p2

Part 2:

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from torch.cuda.amp import GradScaler
#!pip install --upgrade --force-reinstall --no-deps kaggle

Part 3:

## Context 2##
context = 32
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = (2*context+1)*13
        layers = [
            nn.Linear(in_size,512),
            nn.BatchNorm1d(512,affine=False), # NewAdd(1) # Affine=False NA13
            nn.LeakyReLU(),
            nn.Dropout(p=0.1), # NewAdd(2)

            nn.Linear(512,1024), # Earlier it was 40
            nn.BatchNorm1d(1024,affine=False), # NA-14
            nn.LeakyReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(1024,2048), # Earlier it was 40
            nn.BatchNorm1d(2048,affine=False), # NA-14
            nn.ReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(2048,4096), # Earlier it was 40
            nn.BatchNorm1d(4096,affine=False), # NA-14
            nn.ReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(4096,2048), # Earlier it was 40
            nn.BatchNorm1d(2048,affine=False), # NA-14
            nn.ReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(2048,1024), # Earlier it was 40
            nn.BatchNorm1d(1024,affine=False), # NA-14
            nn.ReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(1024,512), # Earlier it was 40
            nn.BatchNorm1d(512,affine=False), # NA-14
            nn.LeakyReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(512,256), # Earlier it was 40
            nn.BatchNorm1d(256,affine=False), # NA-14
            nn.LeakyReLU(),# NA-15
            nn.Dropout(p=0.1),# NA-16

            nn.Linear(256,40),# NA-17
        ]
        self.laysers = nn.Sequential(*layers)

    def forward(self, A0):
        x = self.laysers(A0)
        return x

Part 4:

class LibriSamples(torch.utils.data.Dataset):
    def __init__(self, data_path, sample=20000, shuffle=True, partition="dev-clean", csvpath=None):
        # sample represent how many npy files will be preloaded for one __getitem__ call
        self.sample = sample 
        
        self.X_dir = data_path + "/" + partition + "/mfcc/"
        self.Y_dir = data_path + "/" + partition +"/transcript/"
        
        self.X_names = os.listdir(self.X_dir)
        if (partition=="train-clean-100" or partition=="dev-clean"): 
            self.Y_names = os.listdir(self.Y_dir)

        # using a small part of the dataset to debug
        
        if csvpath:
            if (partition=="train-clean-100" or partition=="dev-clean"):
                subset = self.parse_csv(csvpath)
                self.X_names = [i for i in self.X_names if i in subset]
                self.Y_names = [i for i in self.Y_names if i in subset]
            else:
                self.X_names = list(pd.read_csv(csvpath).file)
                      
        if shuffle == True:
            XY_names = list(zip(self.X_names, self.Y_names))
            random.shuffle(XY_names)
            self.X_names, self.Y_names = zip(*XY_names)
            
        assert(len(self.X_names) == len(self.Y_names))
        self.length = len(self.X_names)
        self.PHONEMES = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
      
    @staticmethod
    def parse_csv(filepath):
        subset = []
        with open(filepath) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                subset.append(row[0])
        return subset[1:]

    def __len__(self):
        return int(np.ceil(self.length / self.sample))
        
    def __getitem__(self, i):
        sample_range = range(i*self.sample, min((i+1)*self.sample, self.length))
                  
        X, Y = [], []
        for j in sample_range:
            X_path = self.X_dir + self.X_names[j]
            Y_path = self.Y_dir + self.Y_names[j]
            
            label = [self.PHONEMES.index(yy) for yy in np.load(Y_path)][1:-1]

            X_data = np.load(X_path)
            X_data = (X_data - X_data.mean(axis=0))/X_data.std(axis=0)
            X.append(X_data)
            Y.append(np.array(label))
        
        X, Y = np.concatenate(X), np.concatenate(Y)
        return X, Y

Part 5:

class LibriItemsTestV2(torch.utils.data.Dataset):
    def __init__(self, data_path, partition, context, LIBRIPATH):
        #assert(X.shape[0] == Y.shape[0])
        df = pd.read_csv(LIBRIPATH)
        
        files = list(df.file)
        
        self.X = []
        
        for t in files:
            r = np.load(data_path + "/" + partition + "/mfcc/"+t)
            r = (r - r.mean(axis=0))/r.std(axis=0)
            self.X.append(r)
        
        self.X=np.concatenate(self.X)
        self.length  = self.X.shape[0]
        self.X = np.pad(self.X,((context,context),(0,0)),mode="constant",constant_values=(0,0))
        
        self.context = context
                    
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
            xx = self.X[i:(2*self.context)+i+1].flatten()
          
            return xx # return xx only for test

Part 6:
Weight Initialization:

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

#net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
model.apply(init_weights)

Part 7:

Implement: def train(args, model, device, train_samples, optimizer, criterion, epoch,num_workers,scaler)
           def test(args, model, device, dev_samples,num_workers)
           def main(args)

Part 8:

device = torch.device("cuda")
model = Network().cuda() # NA 18 cuda()
model.load_state_dict(torch.load("/content/model_epoch_16.pth"))

Part 9:

Implement: def testpred(args, model, device)

Part 10:

Implement: output1 = testpred(args,model,device) 

Part 11:

df=pd.DataFrame()
df['id'] = range(len(output1))
df['label'] = output1
df.to_csv("results.csv",index=False)
! kaggle competitions submit -c 11-785-s22-hw1p2 -f results.csv -m "New Submission"

c] Amendments made overall:

1) Added 1d batchnorm, LeakyReLU, Dropout added in Architecture layers.
2) Created Libri_items test_V2 Function to create library items.
3) Added Mixed Precision Training in test function.
4) Added torch.cuda.empty.chache() in test Function.
5) Added Num_workers as extra parameter in train dataloader.
6) Added LR_Scheduler(ReduceLROnPLateau) in main function.
7) Added weight initialization separately.
8) Created Test Function to Predict output.

D] No.of.Epochs- 50 (stopped at 16 as got high accuracy)
Hyper,Parameters- Batch Size-2048, LR=0.001, log interval:200, CONTEXT=32, Optimizer-Adam, weight initialization- Xavier initilization