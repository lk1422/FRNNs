import torch
import os
import torch.nn as nn
from torch import optim 
from models import RetraNet,Baseline
import json
import hashlib
import dataset
import random
import math
import utils

def write_info(model, metrics, variables, path):
    torch.save(model, path+".pth")
    info = open(path+".txt", 'w')
    info.write(str(variables))
    info.write(str(metrics))
    info.close()

def test_metric(samples, model, crit, dset, device):
    loss = 0
    model.eval()
    for _ in range(samples):
        x,y,y_ = dset.get_batch(1, test=True)
        x = x.type(torch.LongTensor).to(device)
        y = y.type(torch.LongTensor).to(device)
        y_ = y_.type(torch.LongTensor).to(device) 
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = utils.create_mask(x,y,dset, device)
        model_out, _ = model(x, y, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
        
        
        l = crit(model_out.reshape(-1, model_out.shape[-1]), y_.reshape(-1))
        loss+= l.item()
    model.train()

    return loss/samples



def  test_hyperparams(dropout, dims, heads, num_encoders, num_decoders, ff_dims, \
        model_optimizer, model_lr, ac_optimizer, ac_lr, scheduler, iterations, max_val):
    metrics = {"loss":[], "average_loops":[], "final_test": 0}
    device = torch.device('cuda')
    dset = dataset.Arithmetic(max_val, test_data=True )
    model= RetraNet(device=device, actor_optim=ac_optimizer, actor_lr=ac_lr, \
            max_len=dset.max_len, num_tokens=dset.num_tokens, num_encoders=num_encoders, \
            num_decoders=num_decoders, dim=dims, nhead=heads, d_feedforward=ff_dims, \
            dropout=dropout).to(device)
    optimizer = model_optimizer(model.parameters(), model_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=scheduler)
    crit = nn.CrossEntropyLoss(ignore_index=dset.TOKENS["<PAD>"])
    utils.train(model, optimizer, crit, device, iterations, dset, 1 ,10000,scheduler, metrics)
    metrics["final_test"] = test_metric(1000, model, crit, dset, device)
    return model, metrics

def run_tests(model_lr_range, ac_lr_range, schedule_range, dropout_range, dims_range, \
              head_range, num_encoder_range, num_decoder_range, \
              ff_dim_range, max_val, train_iterations, samples):
    for i in range(samples):
        m_lr = random.uniform(model_lr_range[0], model_lr_range[1])
        ac_lr = random.uniform(ac_lr_range[0], ac_lr_range[1])
        sched = random.uniform(schedule_range[0], schedule_range[1])
        dropout = random.uniform(dropout_range[0], dropout_range[1])
        dims =  int(math.pow(2, random.randint(dims_range[0], dims_range[1])))
        heads =  int(math.pow(2, random.randint(head_range[0], head_range[1])))
        encoders = random.randint(num_encoder_range[0], num_encoder_range[1])
        decoders = random.randint(num_decoder_range[0], num_decoder_range[1])
        ff_dim =  int(math.pow(2, random.randint(ff_dim_range[0], ff_dim_range[1])))
        model_optimizer = optim.Adam
        ac_optimizer = optim.Adam

        test_variables = {"model_learning_rate": m_lr, "ac_learning_rate": ac_lr, 
                          "scheduler": sched, "dropout":dropout, "dims":dims, "heads":heads,
                          "encoders":encoders, "decoders":decoders, "ff_dim":ff_dim}

        path = "model_" + str(i) + ".txt"
        test_variables_json = json.dumps(test_variables, indent = 4) 
        path = os.path.join("model_files/frnn", path)
        print(f"ITERATION {i} with variables:")
        print(test_variables_json)

        model, metrics= test_hyperparams(dropout=dropout, dims=dims, heads=heads,\
                num_encoders=encoders, num_decoders=decoders, ff_dims=ff_dim,    \
                model_optimizer=model_optimizer, model_lr=m_lr, ac_optimizer=ac_optimizer, \
                ac_lr=ac_lr, scheduler=sched,iterations=train_iterations, max_val =max_val)
        metrics_json = json.dumps(metrics, indent=4)
        write_info(model, metrics_json, test_variables_json, path)

        

def Baseline_test_values(lr_val, schedule_gamma_val, dropout_val , dim_val, nhead_val, encoder_layers_val, decoder_layers_val, ff_dim_val, optimizer_val, iterations, max_val, batches_val):
    metrics = {"loss":[],  "final_test": 0}
    device = torch.device('cuda')
    dset = dataset.Arithmetic(max_val, test_data=True )
    model = Baseline(device, max_len=dset.max_len, num_tokens=dset.num_tokens,\
            num_encoders=encoder_layers_val, num_decoders=decoder_layers_val, dim=dim_val, nhead=nhead_val,\
            d_feedforward=ff_dim_val, dropout=dropout_val).to(device)
    optimizer = optimizer_val(model.parameters(), lr_val)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=schedule_gamma_val)
    crit = nn.CrossEntropyLoss(ignore_index=dset.TOKENS["<PAD>"])
    utils.train(model, optimizer, crit, device, iterations, dset, batches_val ,10000,scheduler, metrics)
    metrics["final_test"] = test_metric(1000, model, crit, dset, device)

    return model, metrics

def Baseline_test_space(lr, schedule_gamma, dropout, dims, nhead, encoder_layers, decoder_layers, ff_dim, max_val,  iterations, samples, batches):
    models = []
    for i in range(samples):
        """SELECT PARAMETERS"""
        lr_val = random.uniform(lr[0], lr[1])
        schedule_gamma_val = random.uniform(schedule_gamma[0], schedule_gamma[1])
        dropout_val = random.uniform(dropout[0], dropout[1])
        encoder_layers_val = random.randint(encoder_layers[0], encoder_layers[1])
        decoder_layers_val = random.randint(decoder_layers[0], decoder_layers[1])
        ff_dim_val = int(math.pow(2, random.randint(ff_dim[0],ff_dim[1])))
        dims_val   = int(math.pow(2, random.randint(dims[0], dims[1])))
        nhead_val  = int(math.pow(2, random.randint(nhead[0], nhead[1])))
        batches_val= int(math.pow(2, random.randint(batches[0], batches[1])))

        optimizer_val = optim.Adam

        test_variables = { "Baseline":True, "Batch_size":batches_val, "learning_rate": lr_val,  "gamma:":schedule_gamma_val,  "dropout":dropout_val, 
                          "encoder_layers":encoder_layers_val, "decoder_layers":decoder_layers_val,  
                          "feedforward_dim":ff_dim_val, "dims":dims_val, "heads":nhead_val}
        path = hashlib.md5(bytes(str(test_variables),'ascii')).hexdigest()
        test_variables_json = json.dumps(test_variables, indent = 4) 
        path = os.path.join("model_files", path)
        print(f"ITERATION {i} with variables:")
        print(test_variables_json)
        model, metrics = Baseline_test_values(lr_val, schedule_gamma_val, dropout_val, dims_val, nhead_val, encoder_layers_val, \
                    decoder_layers_val, ff_dim_val, optimizer_val, iterations, max_val, batches_val)

        metrics_json = json.dumps(metrics, indent=4)
        write_info(model, metrics_json, test_variables_json, path)
        models.append((path, metrics['final_test']))

        print(f"PREFORMANCE: {metrics['final_test']}")
        print("TEST COMPLETED")
    models = sorted(models, key=lambda x: x[1])
    print(models)

    f = open("model_files/models_sorted.txt",'w')
    f.write(str(models))
    f.close()

if __name__ == "__main__":
    """
def run_tests(model_lr_range, ac_lr_range, schedule_range, dropout_range, dims_range, \
              head_range, num_encoder_range, num_decoder_range, \
              ff_dim_range, max_val, train_iterations, samples):
    """
    LEARNING_RATE = (1e-7, 1e-5)
    LEARNING_RATE2= (1e-7, 1e-5)
    SCHED         = (.90, .99)
    DROPOUT       = (.1,   .3)
    DIMS          = (5,     8)
    HEADS         = (2,     5)
    ENCODER_LAYERS= (2,     4)
    DECODER_LAYERS= (2,     4)
    FF_DIM        = (9,    11)
    MAX_VAL       = 1000
    ITERATIONS    = 200000
    SAMPLES       = 20
    run_tests(LEARNING_RATE, LEARNING_RATE2, SCHED, DROPOUT, DIMS, HEADS, ENCODER_LAYERS, DECODER_LAYERS, FF_DIM, MAX_VAL,\
            ITERATIONS, SAMPLES)

#    Baseline_test_space(LEARNING_RATE, GAMMA, DROPOUT, DIMS, HEADS, ENCODER_LAYERS, DECODER_LAYERS, \
#            FF_DIM, MAX_VAL, ITERATIONS, SAMPLES, BATCHES)
    

       

