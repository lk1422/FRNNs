import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dataset
import math
def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, dset, device):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == dset.TOKENS["<PAD>"])
    tgt_padding_mask = (tgt == dset.TOKENS["<PAD>"])
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def convert(expression:str, dset, model, device):
    ##Convert String to a tensor##
    model.eval()
    src = torch.tensor([dset.tokenize_expression(expression)]).to(device)
    src = src[:, 1:]
    ##Set up output##
    y = torch.ones(1, dset.max_len).fill_(dset.TOKENS["<PAD>"]).type(torch.long).to(device)
    y[0,0] = dset.TOKENS["<SOS>"]
    model.eval()
    for i in range(dset.max_len-1):
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src,y,dset, device)
        out = model(src, y, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
        probs = out[0,i]
        next_token = torch.argmax(probs,dim=0)
        y[0, i+1] = next_token
        if next_token == dset.TOKENS["<EOS>"]:
            break
    y = y.squeeze(0)
    y = y.tolist()
    model.train()
    return dset.get_str(y)

def train(model, optim, crit, device, iterations, dset, batch_size, print_freq, scheduler, metrics):
    running_loss = 0
    for it in range(iterations):
        x,y,y_ = dset.get_batch(batch_size)
        x = x.type(torch.LongTensor).to(device)
        y = y.type(torch.LongTensor).to(device)
        y_ = y_.type(torch.LongTensor).to(device)
        
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(x,y,dset, device)
        model_out, loops= model(x, y, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
        metrics["average_loops"].append(loops)
        
        optim.zero_grad()
        
        loss = crit(model_out.reshape(-1, model_out.shape[-1]), y_.reshape(-1))

        if(model.encoders[0].actor_train):
            model.propagateLoss(loss.item())

        loss.backward()
        
        optim.step()
        running_loss+=loss.item()
        scheduler.step()

        metrics["loss"].append(loss.item())
        
        if (it+1) % print_freq == 0:
            print("Iteration:",it+1,"Loss:",running_loss/print_freq)
            running_loss=0

