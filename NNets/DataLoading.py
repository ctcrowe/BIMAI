import torch
import torch.nn as nn
import time
import os
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

def create_datasets(input_file, dataset):
    with open(input_file, 'r') as f:
        data = f.read()
    inputs = data.splitlines()

    test_set_size = min(1000, int(len(inputs) * 0.1))
    rp = torch.randperm(len(inputs)).tolist()
    train_words = [inputs[i] for i in rp[:-test_set_size]]
    test_words = [inputs[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    train_dataset = dataset(train_words)
    test_dataset = dataset(test_words)
    return train_dataset, test_dataset
  
class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)
    
    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

def evaluate(model, device, dataset, batch_size = 1, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        logits, loss = model(device, *batch)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()
    return mean_loss