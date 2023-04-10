import torch
import torch.nn as nn
import time
import os
import IndexNetwork as indexNetwork
import NNets.DataLoading as dataLoading
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 32
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 6
n_layer = 3
dropout = 0.2
# ------------

txt_path = "Datasets/IndexNetworkData.txt"
path = "Models/IndexNetwork.pt"
model = indexNetwork.IndexModel()
if os.path.isfile(path):
    statedict = torch.load(path)
    model.load_state_dict(statedict)

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

def RunTraining():
    train_dataset, test_dataset = dataLoading.create_datasets(txt_path, indexNetwork.IndexDataset)
    batch_loader = dataLoading.InfiniteDataLoader(train_dataset, batch_size = batch_size)

    best_loss = None
    step = 0

    while True:
        t0 = time.time()
        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        A, B, C, D = batch

        logits, loss = model(device, A, B, C, D)

        model.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        if step > 0 and step % 500 == 0:
            train_loss = dataLoading.evaluate(model, device, train_dataset, batch_size, max_batches=5 * batch_size)
            test_loss = dataLoading.evaluate(model, device, test_dataset, batch_size, max_batches=5 * batch_size)
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                print(f"test loss {test_loss} is the best so far, saving model to {path}")
                torch.save(model.state_dict(), path)
                best_loss = test_loss
            
            
        #if step > 0 and step % 200 == 0:
        #    print_samples(num=10)

        step+=1
        
netType = None
while netType is None:
    netSel = input("Network Type?")
    if netSel == "Index":
        netType = index
        txt_path = "Datasets/IndexNetworkData.txt"
        path = "Models/IndexNetwork.pt"
        model = indexNetwork.IndexModel()
        if os.path.isfile(path):
            statedict = torch.load(path)
            model.load_state_dict(statedict)

        m = model.to(device)
        print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        
while True:
    usage = input("Train or Test?")
    if usage == "Test":
        test = ""
        while test != "X":
            text = input("Test your room name")
            sample = get_Sample(text, True)
            A, B, C, D = sample
            A = A.view(1, -1)
            B = B.view(1, -1)
            C = C.view(1, -1)
            print(A, B)
            logits, loss = model(device, A, B, C)
            print(logits)
            max = torch.argmax(logits)
            print(list(class_map.keys())[max])
    elif usage == "Train":
        RunTraining()
