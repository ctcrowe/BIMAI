import torch
import torch.nn as nn
import NNets.Alpha as Alpha
import NNets.TimedAttention as timed
import NNets.StaticAttention as static
from torch.nn import functional as F
from torch.utils.data import Dataset

# hyperparameters
block_size = 32
embd_size = 32
n_head = 4
input_layers = 2
room_layers = 4
n_layer = 4
dropout = 0.2
# ------------

room_list = ["NULL", "GREAT ROOM", "BEDROOM", "MASTER BEDROOM", "BATHROOM", "MASTER BATHROOM", "POWDER ROOM", "KITCHEN",
             "KITCHEN / LIVING", "LIVING ROOM", "DINING ROOM", "STUDY", "FAMILY ROOM", "OFFICE", "LAUNDRY", "DEN", "CLOSET", "PANTRY"]
data_map = { s:i for i,s in enumerate(room_list) }

encode = lambda s : [data_map[word] for word in s.strip().upper().split(',')[1:]]

def get_Samples(input):
    samples = []
    input_encoding = encode(input)
    input_area = long(input.split(',')[0])
    for i in range(len(input_encoding) - 2):
        samples.append([torcg.tensor(input_area), torch.tensor(input_encoding[:i]), torch.tensor(input_encoding[i+1]))
        
    return samples

def Test(model, text, device):
    sample = get_Sample(text, True)
    A, B, C = sample
    A = A.view(1, -1)
    B = B.view(1, -1)
    logits, loss = model(device, A, B)
    print(logits)
    max = torch.argmax(logits)
    return room_list[max]

class UnitRoomsDataset(Dataset):
    def __init__(self, lines):
        self.data = []
        for line in lines:
            self.data.extend(get_Samples(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        area, rooms, output = self.data[idx]
        return area, rooms, output

class UnitRoomsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_basis = nn.Linear(1, embd_size - 4, dtype = torch.float)
        self.input_blocks = nn.Sequential(*[static.Block(embd_size - 4, n_head = n_head) for _ in range(input_layers)])
        self.room_token_embedding_table = nn.Embedding(len(room_list), embd_size)
        self.room_position_embedding_table = nn.Embedding(block_size, embd_size)
        self.room_blocks = nn.Sequential(*[timed.TimedBlock(2 * embd_size, n_head = n_head) for _ in range(room_layers)])
        self.ln_f = nn.LayerNorm(2 * embd_size)
        self.lm_head = nn.Linear(2 * embd_size, len(room_list))
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, device, X, Y, targets = None):
        B, T = X.shape
        x = self.input_basis(B)
        x = self.input_blocks(x)
        y = torch.rand(n_head, 4, dtype = torch.float)
        x = torch.cat(x, y)
        B, T = Y.shape
        tok_emb = self.room_token_embedding_table(Y)
        pos_emb = self.room_position_embedding_table(torch.arange(T, device = device))
        emb = tok_emb + pos_emb
        x = torch.cat(x, emb)
        x = self.room_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            loss_targets = torch.nn.functional.one_hot(targets, len(room_list))
            loss_targets = loss_targets.view(B, len(room_list.items()))
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss
