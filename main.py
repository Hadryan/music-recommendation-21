import os
import pandas as pd
import torch
import torch.nn as nn
import argparse
import time
import random
from dataloader import Make_Dataset, SampleGenerator
from model import NeuralCF
from evaluate import Engine
from metrics import MetronAtK
import wandb
import warnings

warnings.filterwarnings("ignore")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim',
                type=str,
                default='adam',
                help='optimizer')
    parser.add_argument('--lr',
                type=float,
                default=0.001,
                help='learning rate')
    parser.add_argument('--epochs',
                type=int,
                default=100,
                help='learning rate')
    parser.add_argument('--batch_size',
                type=int,
                default=1024,
                help='train batch size')
    parser.add_argument('--latent_dim_mf',
                type=int,
                default=8,
                help='latent_dim_mf')
    parser.add_argument('--num_layers',
                type=int,
                default=3,
                help='num layers')
    parser.add_argument('--num_neg',
                type=int,
                default=10,
                help='negative sample')
    parser.add_argument('--l2',
                type=float,
                default=0.0,
                help='l2_regularization')
    parser.add_argument('--gpu',
                type=str,
                default='0',
                help='gpu number')
    args = parser.parse_args()

    # wandb 
    wandb.init(project="recommendation")
    wandb.config.update(args)
    
    # define gpu device 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load the data 
    data = pd.read_feather("melon_35919.ftr")
    
    MD = Make_Dataset(ratings = data)
    user, item, rating = MD.trainset
    evaluate_data = MD.evaluate_data


    # load the model(ncf)
    model = NeuralCF(num_users= len(data),num_items = 35919, 
                     embedding_size = args.latent_dim_mf,
                     num_layers = args.num_layers)
    model.cuda()
    model = nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()
    
    wandb.watch(model)

    N = []
    patience = 0
    for epoch in range(args.epochs):
        print('Epoch {} starts !'.format(epoch+1))
        print('-' * 80)
        t1 = time.time()
        model.train()
        total_loss = 0
        sample = SampleGenerator(user = user, item = item, 
                                 rating = rating, ratings = data, 
                                 positive_len = MD.positive_len, num_neg = args.num_neg)
        train_loader = sample.instance_a_train_loader(args.batch_size)
        print("Generate train data!")

        for batch_id, batch in enumerate(train_loader):
            users, items, ratings = batch[0], batch[1], batch[2]
            ratings = ratings.float()
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            optimizer.zero_grad()
            output = model(users, items)
            loss = criterion(output, ratings)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            wandb.log({'Batch Loss': loss})
            total_loss += loss

        t2 = time.time()
        print("train : ", t2 - t1) 
 

        # define engine
        engine = Engine()
        hit_ratio,ndcg = engine.evaluate(model,evaluate_data, epoch_id=epoch)
        wandb.log({"epoch" : epoch,
                    "HR" : hit_ratio,
                    "NDCG" : ndcg})
        N.append(ndcg)

        if N[-1] < max(N):
            if patience > 10:
                print("Patience = 10 over")
                print("ndcg = {:.4f}".format(max(N)))
                break
            else:
                patience += 1
                print("Patience = {} ndcg = {:.4f}".format(patience, max(N)))
        else:
            patience = 0
            print("Patience = {}".format(patience))

    
if __name__ == '__main__':
    
    file_name = "melon_35919.ftr"
    
    if os.path.exists(file_name): 
        main()
    
        
        