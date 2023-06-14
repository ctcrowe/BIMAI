import torch
import torch.nn as nn
import NNets.Alpha as Alpha
import NNets.TimedAttention as timed
import NNets.StaticAttention as static
from torch.nn import functional as F
from torch.utils.data import Dataset

# hyperparameters
external_block_size = 256
internal_block_size = 8
half_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2
# ------------



def get_Sample(input, printSample=False):
    input = input.strip().upper()
    lines = input.split(',')
    line = lines[0]
    line2 = lines[1]
    sample = [0] * external_block_size
    catName = [0] * external_block_size
    cutSurf = [0]
    for i in range(external_block_size):
        try : sample[i] = Alpha.chars.index(line[i])
        except : pass
    for i in range(external_block_size):
        try : catName[i] = Alpha.chars.index(line2[i])
        except : pass
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
    return str(max.item())

class VisibilityDataset(Dataset):
    def __init__(self, lines):
        self.data = []
        self.chars = Alpha.chars
        self.max_len = external_block_size
        for line in lines:
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
        self.name_position_embedding_table = nn.Embedding(external_block_size, half_embd) #replace with internal_block_size when prepared
        self.cat_token_embedding_table = nn.Embedding(len(Alpha.chars), half_embd)
        self.cat_position_embedding_table = nn.Embedding(external_block_size, half_embd) #replace with internal_block_size when prepared
        self.cutSurf_head = nn.Linear(1, half_embd, dtype = torch.float)
        self.first_block = timed.TimedBlock(2 * half_embd, n_head=n_head, block_size=external_block_size) #replace with internal_block_size when prepared
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

    def FirstBlockPass(self, device, A, B):
        Batch1, T1 = A.shape
        name_tok_emb = self.name_token_embedding_table(A)
        name_pos_emb = self.name_position_embedding_table(torch.arange(T1, device = device)) #shape is 64, 64
        Batch2, T2 = B.shape #shape is 8, 64
        cat_tok_emb = self.cat_token_embedding_table(B) #shape is 8, 64, 64
        cat_pos_emb = self.cat_position_embedding_table(torch.arange(T2, device = device)) #shape is 64, 64

        name = name_tok_emb + name_pos_emb #shape is 8, 256, 32
        cat = cat_tok_emb + cat_pos_emb #shape is 8, 256, 32
        x = torch.cat([name, cat], dim=-1)
        x = self.first_block(x)
        return x
    
    def forward(self, device, A, B, C, targets = None):
        x = self.FirstBlockPass(device, A, B) #shape is 8, 64, 64
        x = torch.sum(x, dim=-2, keepdim=False) #shape is 8, 128

        cutSurf_x = self.cutSurf_head(C)
        x = torch.cat([x, cutSurf_x], dim=-1)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #shape is 8, 17

        if targets is None:
            loss = None
        else:
            Y, Z = logits.shape
            logits = logits.view(Y, Z)
            loss_targets = torch.nn.functional.one_hot(targets, 17)
            loss_targets = loss_targets.view(Y, 17)
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss
