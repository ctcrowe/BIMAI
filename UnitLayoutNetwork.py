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

def get_Sample(area, encoding, end_vector_point):
    dataset = [0] * block_size
    input_area = [(float)(input.split(',')[0])]
    for i in range(len(input_encoding) - 2):
        dataset = [0] * block_size
        for j in range(block_size):
            try : dataset[j] = input_encoding[i + j]
            except : dataset[j] = 0
        samples.append([torch.tensor(input_area), torch.tensor(dataset), torch.tensor(input_encoding[i+1])])

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
            input_area = [(float)(line.split(',')[0])]
            input_encoding = encode(line)
            for i in range(len(input_encoding) - 2):
                self.data.append(get_Sample(input_area, input_encoding, i))
    
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
        self.room_blocks = nn.Sequential(*[timed.TimedBlock(2 * embd_size, n_head = n_head, block_size = block_size) for _ in range(room_layers)])
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

    def forward(self, device, X, Y, targets = None, batch_size = 8):
        x = self.input_basis(X)
        x = self.input_blocks(x)
        y = torch.rand(batch_size, 4, dtype = torch.float)
        x = torch.cat([x, y], dim=-1)
        B, T = Y.shape
        tok_emb = self.room_token_embedding_table(Y)
        pos_emb = self.room_position_embedding_table(torch.arange(T, device = device))
        emb = tok_emb + pos_emb
        x = x.unsqueeze(dim = -2).expand(-1, embd_size, -1)
        x = torch.cat([x, emb], dim=-1)
        x = self.room_blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        logits = torch.sum(x, dim=-2, keepdim=False)

        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            loss_targets = torch.nn.functional.one_hot(targets, len(room_list))
            loss_targets = loss_targets.view(B, len(room_list))
            loss = F.cross_entropy(logits, loss_targets.type(torch.FloatTensor))

        return logits, loss
