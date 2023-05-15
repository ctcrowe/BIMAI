import torch
import torch.nn as nn
import NNets.Alpha as Alpha
import NNets.TimedAttention as timed
import NNets.StaticAttention as static
from torch.nn import functional as F
from torch.utils.data import Dataset

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 64
max_iters = 5000
eval_interval = 100
eval_iters = 200
half_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

def get_Sample(input, printSample=False):
    input = input.strip().upper()
    lines = input.split(',')
    line = lines[0]
    line2 = lines[1]
    sample = [0] * block_size
    catName = [0] * block_size
    cutSurf = [0]
    for i in range(len(line)):
        try :
            sample[i] = Alpha.chars.index(line[i])
        except :
            pass
    for i in range(len(line2)):
        try :
            catName[i] = Alpha.chars.index(line2[i])
        except :
            pass
    try :
        cutSurf[0] = int(lines[2]) - 1
    except :
        cutSurf[0] = 0
    try :
        classification = int(lines[-1])
    except :
        classification = 1
        
    if printSample :
        print(input, sample, catName, cutSurf, classification)
    return torch.tensor(sample), torch.tensor(catName), torch.tensor(cutSurf, dtype = torch.float), torch.tensor(classification)

def Test(model, text, device):
    sample = get_Sample(text, True)
    A, B, C, D = sample
    A = A.view(1, -1)
    B = B.view(1, -1)
    C = C.view(1, -1)
    print(A, B)
    logits, loss = model(device, A, B, C)
    print(logits)
    max = torch.argmax(logits)
    return max

class VisibilityDataset(Dataset):
    def __init__(self, lines):
        self.data = []
        self.chars = Alpha.chars
        self.max_len = block_size
        for line in lines: # text.splitlines():
            name, category, gtype, sample = get_Sample(line)
            self.data.append([name, category, gtype, sample])
        self.stoi = {ch:i+1 for i,ch in enumerate(Alpha.chars)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        name, category, gtype, output = self.data[idx]
        return name, category, gtype, output
    
class VisibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name_token_embedding_table = nn.Embedding(len(Alpha.chars), half_embd)
        self.name_position_embedding_table = nn.Embedding(block_size, half_embd)
        self.cat_token_embedding_table = nn.Embedding(len(Alpha.chars), half_embd)
        self.cat_position_embedding_table = nn.Embedding(block_size, half_embd)
        self.cutSurf_head = nn.Linear(1, half_embd, dtype = torch.float)
        self.first_block = timed.TimedBlock(2 * half_embd, n_head=n_head, block_size=block_size)
        self.blocks = nn.Sequential(*[static.Block(3 * half_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(3 * half_embd)
        self.lm_head = nn.Linear(3 * half_embd, 17)
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, device, A, B, C, targets = None):
        Batch1, T1 = A.shape
        tok_emb1 = self.name_token_embedding_table(A)
        pos_emb1 = self.name_position_embedding_table(torch.arange(T1, device = device))
        Batch2, T2 = B.shape
        tok_emb2 = self.cat_token_embedding_table(B)
        pos_emb2 = self.cat_position_embedding_table(torch.arange(T2, device = device))
        x1 = tok_emb1 + pos_emb1
        x2 = tok_emb2 + pos_emb2
        x = torch.cat([x1, x2], dim=-1)
        x = self.first_block(x)
        x = torch.sum(x, dim=-2, keepdim=False)

        cutSurf_x = self.cutSurf_head(C)
        x = torch.cat([x, cutSurf_x], dim=-1)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            Y, Z = logits.shape
            logits = logits.view(Y, Z)
            loss_targets = torch.nn.functional.one_hot(targets, 17)
            loss_targets = loss_targets.view(Y, 17)
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss
