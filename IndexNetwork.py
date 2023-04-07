import torch
import torch.nn as nn
import time
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

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
