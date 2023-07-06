import torch
import torch.nn as nn
import NNets.Alpha as Alpha
import NNets.TimedAttention as timed
import NNets.StaticAttention as static
from torch.nn import functional as F
from torch.utils.data import Dataset

# hyperparameters
block_size = 32
embd_size = 128
n_head = 4
input_layers = 2
room_layers = 4
n_layer = 4
dropout = 0.2
# ------------

room_list = ["NULL", "GREAT ROOM", "BEDROOM", "MASTER BEDROOM", "BATHROOM", "MASTER BATHROOM", "POWDER ROOM", "KITCHEN",
             "KITCHEN / LIVING", "LIVING ROOM", "DINING ROOM", "STUDY", "FAMILY ROOM", "OFFICE", "LAUNDRY", "DEN", "CLOSET", "PANTRY"]
data_map = { s:i for i,s in enumerate(room_list) }

encode = lambda s : [data_map[word] for word in s]

def get_Sample(area, encoding, desired_areas = None):
    dataset = [0] * block_size
    output = [0] * block_size
    input_area = [(float)(area)]
    for i in range(encoding):
        try : dataset[i] = encoding[i]
        except : dataset[i] = 0
        if desired_areas is not None:
            try : output[i] = desired_areas[i] / area
            except : output[i] = 0
    return torch.tensor(input_area), torch.tensor(dataset), torch.tensor(output)

def Test(model, text, device):
    input = text.split(',')
    encoding = encode(input[1:])
    size = (float)(input[0])
    sample = get_Sample(size, encoding)
    A, B, C = sample
    A = A.view(1, -1)
    B = B.view(1, -1)
    logits, loss = model(device, A, B)
    return logits

class UnitRoomsDataset(Dataset):
    def __init__(self, lines):
        self.data = []
        for line in lines:
            input = line.split(',')
            input_size = (len(input) - 1) / 2
            input_area = (input[0])
            input_encoding = encode(input[1:input_size + 1])
            output = input[input_size + 1:]
            self.data.append(get_Sample(input_area, input_encoding, output))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        area, rooms, output = self.data[idx]
        return area, rooms, output

class UnitRoomsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_basis = nn.Linear(1, embd_size, dtype = torch.float)
        self.room_token_embedding_table = nn.Embedding(len(room_list), embd_size)
        self.room_position_embedding_table = nn.Embedding(block_size, embd_size)
        self.room_blocks = nn.Sequential(*[timed.TimedBlock(embd_size, n_head = n_head, block_size = block_size) for _ in range(input_layers)])
        self.blocks = nn.Sequential(*[static.Block(2 * embd_size, n_head = n_head) for _ in range(room_layers)])
        self.ln_f = nn.LayerNorm(2 * embd_size)
        self.lm_head = nn.Linear(2 * embd_size, block_size)
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, device, X, Y, targets = None):
        x = self.input_basis(X)
        B, T = Y.shape
        tok_emb = self.room_token_embedding_table(Y)
        pos_emb = self.room_position_embedding_table(torch.arange(T, device = device))
        emb = tok_emb + pos_emb
        emb = self.room_blocks(emb)
        emb = torch.sum(emb, dim=-2, keepdim = False)
        x = torch.cat([emb, x], dim = -1)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        logits = F.softmax(x, dim=-1)

        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            loss_targets = torch.nn.functional.one_hot(targets, len(room_list))
            loss_targets = loss_targets.view(B, len(room_list))
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss