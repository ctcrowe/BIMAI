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

class_map = {"Accessory Storage Areas, Mechanical Equipment Room" : 0, "Agricultural Building" : 1, "Aircraft Hangars" : 2, "Baggage Claim" : 3, "Baggage Handling" : 4,
            "Airport Concourse" : 5, "Airport Waiting Areas" : 6, "Gaming Floors" : 7, "Exhibit Gallery and Museum" : 8, "Assembly Fixed Seats" : 9,
             "Assembly Concentrated" : 10, "Assembly Standing" : 11, "Assembly Unconcentrated" : 12, "Bowling Additional Areas" : 13, "Business Areas" : 14,
            "Business Concentrated" : 15, "Courtrooms" : 16, "Day Care" : 17, "Dormitories" : 18, "Classroom" : 19, "Shops and Other Vocational Areas" : 20,
            "Exercise Rooms" : 21, "Group H-5 Fabrication and Manufacturing Areas" : 22, "Industrial Areas" : 23, "Inpatient Treatment Areas" : 24, "Outpatient Areas" : 25,
            "Institutional Sleeping Areas" : 26, "Commercial Kitchens" : 27, "Educational Laboratory" : 28, "Laboratory" : 29, "Laboratory Suite" : 30, "Library Reading Room" : 31,
            "Library Stack" : 32, "Locker Rooms" : 33, "Mall Buildings" : 34, "Mercantile" : 35, "Storage Stock and Shipping Areas" : 36, "Parking Garages" : 37, "Residential" : 38,
            "Skating Rinks and Swimming Pools" : 39, "Rink and Pool Decks" : 40, "Stages and Platforms" : 41, "Warehouses" : 42}

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
    print(list(class_map.keys())[max])

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
