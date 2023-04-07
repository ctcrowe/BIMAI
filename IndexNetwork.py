import torch
import torch.nn as nn
import time
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 6
n_layer = 3
dropout = 0.2
# ------------

class_map = {"0 - GENERAL" : 0, "1 - SITE INFORMATION" : 1, "2 - BUILDING PLANS" : 2, "3 - BUILDING ELEVATIONS" : 3, "4 - ENLARGED VIEWS": 4,
             "5 - WALL SECTIONS AND ELEVATIONS" : 5, "6 - PARTITION TYPES LEGENDS AND SCHEDULES" : 6, "7 - VERTICAL CIRCULATION" : 7,
             "8 - EXTERIOR DETAILS" : 8, "9 - INTERIOR DETAILS" : 9,  "D - DEMOLITION" : 10}

def get_Sample(input, printSample=False):
    input = input.strip().upper()
    lines = input.split(',')
    line = lines[0]
    sample = [0] * block_size
    size = [0]
    types = [0]
    for i in range(len(line)):
        try :
            sample[i] = chars.index(line[i])
        except :
            pass

    try :
        size[0] = (float)(lines[1])
    except: size[0] = 48
    
    try :
        types[0] = (float)(lines[2])
    except: types[0] = 0
    
    try :
        classification = lines[-1]
        classification = class_map[classification]
    except :
        classification = 0

    if printSample :
        print(input, sample, classification)
    return torch.tensor(sample), torch.tensor(size, dtype = torch.float), torch.tensor(types, dtype = torch.float), torch.tensor(classification)

  class IndexDataset(Dataset):
    def __init__(self, lines):
        #self.txt_path = "/workspaces/OLF-Data/OLFNetworkData.txt"
        self.data = []
        self.chars = chars
        self.class_map = class_map
        self.max_len = block_size
        #with open('OLFNetworkData.txt', 'r', encoding='utf-8') as f:
            #text = f.read()
        for line in lines: # text.splitlines():
            name, view_size, view_type, sample = get_Sample(line)
            self.data.append([name, view_size, view_type, sample])
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        name, view_size, view_type, sample = self.data[idx]
        return name, view_size, view_type, sample
