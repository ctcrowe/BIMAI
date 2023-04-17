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
learning_rate = 3e-4
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

class_map = {"0" : 0, "5" : 1, "7" : 2, "11" : 3, "15" : 4, "20" : 5, "30" : 6, "35" : 7, "40" : 8, "50" : 9, "60" : 10, "100" : 11, "120" : 12,
            "150" : 13, "200" : 14, "240" : 15, "300" : 16, "500" : 17}

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

class OLFDataset(Dataset):
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
    
class OLFModel(nn.Module):
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