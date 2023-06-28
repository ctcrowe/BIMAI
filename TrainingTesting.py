import torch
import time
import os
import NNets.DataLoading as dataLoading
from PickModel import pick

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
learning_rate = 3e-6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

def RunTraining(dataset):
    train_dataset, test_dataset = dataLoading.create_datasets(txt_path, dataset)
    batch_loader = dataLoading.InfiniteDataLoader(train_dataset, batch_size = batch_size)

    best_loss = None
    step = 0

    while True:
        t0 = time.time()
        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]

        logits, loss = model(device, *batch)

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

        step+=1
        
while True:
    datatype = input("Network Type?")
    if datatype != "Unit":
        txt_path, path, model, testMdl, dataset = pick(datatype)
        if model is not None:
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
                        print(testMdl(model, text, device))
                elif usage == "Train":
                    RunTraining(dataset)
    else:
        break
