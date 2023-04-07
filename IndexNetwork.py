import torch
import torch.nn as nn
import time
import os
import NNets.Alpha as Alpha
import NNets.TimedAttention as timed
import NNets.StaticAttention as static
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
            sample[i] = Alpha.chars.index(line[i])
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
        self.chars = Alpha.chars
        self.class_map = class_map
        self.max_len = block_size
        #with open('OLFNetworkData.txt', 'r', encoding='utf-8') as f:
            #text = f.read()
        for line in lines: # text.splitlines():
            name, view_size, view_type, sample = get_Sample(line)
            self.data.append([name, view_size, view_type, sample])
        self.stoi = {ch:i+1 for i,ch in enumerate(Alpha.chars)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        name, view_size, view_type, sample = self.data[idx]
        return name, view_size, view_type, sample

class IndexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(Alpha.chars), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.size_head = nn.Linear(1, n_embd, dtype = torch.float)
        self.type_head = nn.Linear(1, n_embd, dtype = torch.float)
        self.first_blocks = nn.Sequential(*[timed.TimedBlock(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        self.blocks = nn.Sequential(*[static.Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, len(class_map.items()))
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, A, B, C, targets = None):
        Batch, T = A.shape
        tok_emb = self.token_embedding_table(A)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        size_emb = self.size_head(B)
        type_emb = self.type_head(C)
        x = tok_emb + pos_emb
        x = self.first_blocks(x)
        x = torch.sum(x, dim=-2, keepdim = False)
        x = x + type_emb + size_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            loss_targets = torch.nn.functional.one_hot(targets, len(class_map.items()))
            loss_targets = loss_targets.view(B, len(class_map.items()))
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss
