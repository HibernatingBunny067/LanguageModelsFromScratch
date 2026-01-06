import torch
from NanoGPT import NanoGPT
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from utils import get_batch,estimate_loss,custom_tokenizer
torch.manual_seed(42)

@dataclass
class config:
    n_layer = 8
    block_size = 16
    embed = 512
    n_heads = 16
    batch_size = 4
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    max_iters_training = 5000
    lr = 1e-3
    eval_iters = 300
    data_path = r'data/tiny_shakespeare.txt'

tokenizer = custom_tokenizer(data_path=config.data_path)
train_data,val_data = tokenizer.get_coded_data_split()

model = NanoGPT(config,tokenizer._size_())
model.to(config.device)

optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr)

print('Starting Training....')
for iter in range(config.max_iters_training):

    if iter% config.eval_iters == 0:
        losses = estimate_loss(model,train_data,val_data,config.eval_iters,config.block_size,config.batch_size,config.device)
        print(f'step {iter+1}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}')

    X,y = get_batch(train_data,config.block_size,config.batch_size,config.device)
    logits,loss = model(X,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Trained model output:\n")
context = torch.zeros((1,1),dtype=torch.long,device=config.device)
print(tokenizer.decode(model.generate(context,500).squeeze(0).tolist()))