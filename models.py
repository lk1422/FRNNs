import torch
from actor_critic import PolicyNetwork
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import dataset
import math

"""
########################################################################################
BASIC RECCURENT BLOCKS
########################################################################################
"""
REPEAT = 0
TERMINAL = 1
class RetraEncBlock(nn.Module):
    def __init__(self,device, d_model, nhead, actor_optim, actor_lr, actor_train=True, GAMMA=0.99, max_runtime=3, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super(RetraEncBlock, self).__init__()
        self.encoder_block = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, \
                                                        dropout=dropout, batch_first=batch_first)
        self.actor_critic = PolicyNetwork(d_model, 2, device)
        self.actor_optim = actor_optim(self.actor_critic.parameters(), actor_lr)
        self.GAMMA = GAMMA
        self.actor_train = actor_train
        
        self.max_runtime = max_runtime
    
    def propagateLoss(self, loss):
        ##Take the negative of the loss so we minimize the loss
        assert(self.actor_train)
        self.actor_critic.addReward(loss)
        self.actor_critic.train(self.actor_optim, self.GAMMA)
        
    def forward(self, x, mask, pad_mask):
        action = REPEAT
        i=0
        while action == REPEAT and i < self.max_runtime:
            x = self.encoder_block(x, mask, pad_mask)
            
            
            if not self.actor_train:
                break
                
            a = self.actor_critic.select_action(x[:,0])
            if a == REPEAT:
                self.actor_critic.addReward(0)
            
                
            i+=1
        return x, i
            
class RetraDecBlock(nn.Module):
    def __init__(self,device, d_model, nhead, actor_optim, actor_lr,  actor_train=True,  GAMMA=0.99, max_runtime=3, dim_feedforward=2048, dropout=0.1, batch_first=True):
        super(RetraDecBlock, self).__init__()
        self.decoder_block = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, \
                                                        dropout=dropout, batch_first=batch_first)
        self.actor_critic = PolicyNetwork(d_model, 2, device)
        self.actor_optim = actor_optim(self.actor_critic.parameters(), 1e-5)
        self.GAMMA = GAMMA
        self.actor_train = actor_train
        
        
        self.max_runtime = max_runtime
        
    def propagateLoss(self, loss):
        ##Take the negative of the loss so we minimize the loss
        assert(self.actor_train)
        self.actor_critic.addReward(loss)
        self.actor_critic.train(self.actor_optim, self.GAMMA)
        
    def forward(self, x, mem, mask, pad_mask):
        action = REPEAT
        i=0
        while action == REPEAT and i < self.max_runtime:
            x = self.decoder_block(x, mem, tgt_mask=mask, tgt_key_padding_mask=pad_mask)
            
            
            if not self.actor_train:
                break
                
            a = self.actor_critic.select_action(x[:,0])
            if a == REPEAT:
                self.actor_critic.addReward(0)
                
            i+=1
        return x, i

                       
"""
########################################################################################
FULL MODELS
########################################################################################
"""
class RetraNet(nn.Module):
    def __init__(self, device,
                    actor_optim,
                    actor_lr, 
                    max_len,
                    num_tokens,
                    num_encoders=1,
                    num_decoders=1,
                    dim=64,
                    nhead=8,
                    d_feedforward=1024,
                    dropout=0.3
                 ):
        super(RetraNet, self).__init__()
        self.max_len = max_len
        self.device = device
        self.tokens = num_tokens
        ##Create encoder layers##
        self.src_emb = nn.Embedding(num_tokens, dim)
        self.tgt_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        ##Create Transformer##
        self.encoders = [RetraEncBlock(device, dim, nhead, actor_optim, actor_lr, dim_feedforward=d_feedforward, dropout=dropout).to(device) for _ in range(num_encoders)]
        
        self.decoders = [RetraDecBlock(device, dim,nhead, actor_optim, actor_lr, dim_feedforward=d_feedforward, dropout=dropout).to(device) for _ in range(num_decoders)]
        ##Create Final Linear Layer##
        self.linear = nn.Linear(dim, num_tokens)
        ##Create TimeStep Input##
        self.timesteps = torch.Tensor([[i for i in range(max_len)]]).type(torch.LongTensor).to(device)
    def forward(self, src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask):
        pos_emb = self.pos_emb(self.timesteps)
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        mem = pos_emb + src_emb
        out = pos_emb + tgt_emb
        iterations = 0
        for enc in self.encoders:
            mem, i = enc(mem, src_mask, src_pad_mask)
            iterations += i
        for dec in self.decoders:
            out, i = dec(out, mem, tgt_mask, tgt_pad_mask)
            iterations += i
        
        return (self.linear(out)), (iterations/(len(self.encoders) + len(self.decoders)))
    
    def propagateLoss(self, loss):
        for enc in self.encoders:
            enc.propagateLoss(loss)
        for dec in self.decoders:
            dec.propagateLoss(loss)
    def toggleActor(self):
        for enc in self.encoders:
            enc.actor_train = not enc.actor_train
        for dec in self.decoders:
            dec.actor_train = not dec.actor_train
        
        
        

        
        
class Baseline(nn.Module):
    def __init__(self, device,
                    max_len,
                    num_tokens,
                    dim=64,
                    nhead=8,
                    num_encoders=2,
                    num_decoders=2,
                    d_feedforward=1024,
                    dropout=0.1,
                    batch_first=True):
        super(Baseline, self).__init__()
        self.max_len = max_len
        self.device = device
        self.tokens = num_tokens
        ##Create encoder layers##
        self.src_emb = nn.Embedding(num_tokens, dim)
        self.tgt_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        ##Create Transformer##
        self.transformer = nn.Transformer(d_model=dim, nhead=nhead,num_encoder_layers=num_encoders,     \
                                         num_decoder_layers=num_decoders, dim_feedforward=d_feedforward, \
                                         dropout=dropout, batch_first=batch_first)
        ##Create Final Linear Layer##
        self.linear = nn.Linear(dim, num_tokens)
        ##Create TimeStep Input##
        self.timesteps = torch.Tensor([[i for i in range(max_len)]]).type(torch.LongTensor).to(device)
    def forward(self, src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, metrics=None):
        pos_emb = self.pos_emb(self.timesteps)
        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        src_in = pos_emb + src_emb
        tgt_in = pos_emb + tgt_emb
        trans_out = self.transformer(src_in, tgt_in, src_mask, tgt_mask, None, src_pad_mask, tgt_pad_mask)
        return (self.linear(trans_out))
        
        
