import torch
import torch.nn as nn
import NNets.Alpha as Alpha
import NNets.TimedAttention as timed
from torch.nn import functional as F
from torch.utils.data import Dataset

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 128
max_iters = 5000
eval_interval = 100
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

class_map = {"ACCESSORY STORAGE AREAS AND MECHANICAL EQUIPMENT ROOM" : 0,
             "AGRICULTURAL BUILDING" : 1,
             "AIRCRAFT HANGARS" : 2,
             "BAGGAGE CLAIM" : 3,
             "BAGGAGE HANDLING" : 4,
             "AIRPORT CONCOURSE" : 5,
             "AIRPORT WAITING AREAS" : 6,
             "GAMING FLOORS" : 7,
             "EXHIBIT GALLERY AND MUSEUM" : 8,
             "ASSEMBLY FIXED SEATS" : 9,
             "ASSEMBLY CONCENTRATED" : 10,
             "ASSEMBLY STANDING" : 11,
             "ASSEMBLY UNCONCENTRATED" : 12,
             "BOWLING ADDITIONAL AREAS" : 13,
             "BUSINESS AREAS" : 14,
             "BUSINESS CONCENTRATED" : 15,
             "COURTROOMS" : 16,
             "DAY CARE" : 17,
             "DORMITORIES" : 18,
             "CLASSROOM" : 19,
             "SHOPES AND OTHER VOCATIONAL AREAS" : 20,
             "EXERCISE ROOMS" : 21,
             "GROUP H-5 FABRICATION AND MANUFACTURING AREAS" : 22,
             "INDUSTRIAL AREAS" : 23,
             "INPATIENT TREATMENT AREAS" : 24,
             "OUTPATIENT AREAS" : 25,
             "INSTITUTIONAL SLEEPING AREAS" : 26,
             "COMMERCIAL KITCHENS" : 27,
             "EDUCATIONAL LABORATORY" : 28,
             "LABORATORY" : 29,
             "LABORATORY SUITE" : 30,
             "LIBRARY READING ROOM" : 31,
             "LIBRARY STACK" : 32,
             "LOCKER ROOMS" : 33,
             "MALL BUILDINGS" : 34,
             "MERCANTILE" : 35,
             "STORAGE STOCK AND SHIPPING AREAS" : 36,
             "PARKING GARAGES" : 37,
             "RESIDENTIAL" : 38,
             "SKATING RINKS AND SWIMMING POOLS" : 39,
             "RINK AND POOL DECKS" : 40,
             "STAGES AND PLATFORMS" : 41,
             "WAREHOUSES" : 42,
             "N/A" : 43}

def get_Sample(input, printSample=False):
    input = input.strip().upper()
    lines = input.split(',')
    line = lines[0]
    sample = [0] * block_size
    for i in range(len(line)):
        try :
            sample[i] = Alpha.chars.index(line[i])
        except :
            pass
    try :
        classification = lines[-1]
        classification = class_map[classification]
    except :
        classification = 0
        
    if printSample :
        print(input, sample, classification)
    return torch.tensor(sample), torch.tensor(classification)

def Test(model, text, device):
    sample = get_Sample(text, True)
    A, B = sample
    A = A.view(1, -1)
    print(A)
    logits, loss = model(device, A)
    print(logits)
    max = torch.argmax(logits)
    return list(class_map.keys())[max]

class SpaceFunctionDataset(Dataset):
    def __init__(self, lines):
        self.data = []
        self.chars = Alpha.chars
        self.class_map = class_map
        self.max_len = block_size
        #with open('OLFNetworkData.txt', 'r', encoding='utf-8') as f:
            #text = f.read()
        for line in lines: # text.splitlines():
            base, sample = get_Sample(line)
            self.data.append([base, sample])
        self.stoi = {ch:i+1 for i,ch in enumerate(Alpha.chars)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, class_name = self.data[idx]
        return data, class_name
    
class SpaceFunctionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(Alpha.chars), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[timed.TimedBlock(n_embd, n_head = n_head,block_size=block_size) for _ in range(n_layer)])
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
    
    def forward(self, device, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        logits = torch.sum(x, dim=-2, keepdim=False)

        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            loss_targets = torch.nn.functional.one_hot(targets, len(class_map.items()))
            loss_targets = loss_targets.view(B, len(class_map.items()))
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss
