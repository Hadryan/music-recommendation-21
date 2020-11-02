import os
import pandas as pd
import torch
import torch.nn as nn
import argparse
import time
import random
from preprocess import to_date,pre
from model import NeuralCF
from dataloader import SampleGenerator
from evaluate import Engine
from metrics import MetronAtK
import matplotlib.pyplot as plt
import wandb


def main():
    wandb.init(project="recommendation")
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
                default=10,
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
    parser.add_argument('--num_ng',
                type=int,
                default=4,
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
    wandb.config.update(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    train = pd.read_json("/daintlab/data/music_rec/train.json")
    song_meta = pd.read_json("/daintlab/data/music_rec/song_meta.json")
    data,n_data,n_songs=pre(train,song_meta,args.num_ng)


    # users,songs & Reindex
    print('User: {} | Song : {}'.format(n_data,n_songs))
    print('Range of userId is [{}, {}]'.format(data["userId"].min(), data["userId"].max()))
    print('Range of itemId is [{}, {}]'.format(data["itemId"].min(), data["itemId"].max()))
    
    # DataLoader for training
    print("SampleGenerator")
    t1 = time.time()
    sample_generator = SampleGenerator(ratings=data)
    t2 = time.time()
    print("SamplesGenerator Time : ",t2-t1)

    print("EvaluateGenerator")
    t1 = time.time()
    evaluate_data = sample_generator.evaluate_data
    t2 = time.time()
    print("EvaluateGenerator Time : ", t2 - t1)

    #NCF model
    model = NeuralCF(n_data,n_songs,args.latent_dim_mf,args.num_layers)
    model = nn.DataParallel(model).cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
    criterion = nn.BCEWithLogitsLoss()
    wandb.watch(model)


    t1 = time.time()
    train_loader = sample_generator.instance_a_train_loader(args.batch_size)
    t2 = time.time()
    print("train_loader : ", t2 - t1)

    for epoch in range(args.epochs):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        t1 = time.time()
        random.shuffle(train_loader.sampler.seq)
        t2 = time.time()
        print("Data Loader : ", t2 - t1)
        t1 = time.time()
        model.train()
        total_loss = 0
        count=0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            users, items, ratings = batch[0], batch[1], batch[2]
            ratings = ratings.float()
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            optimizer.zero_grad()
            output = model(users, items).cuda()
            loss = criterion(output, ratings)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            total_loss += loss
            count+=1
        t2 = time.time()
        print("train : ", t2 - t1) 
        t1 = time.time()   
        engine = Engine()
        hit_ratio,ndcg = engine.evaluate(model,evaluate_data, epoch_id=epoch)
        wandb.log({"epoch" : epoch,
                    "Loss" : total_loss/count,
                    "HR" : hit_ratio,
                    "NDCG" : ndcg})
        t2 = time.time()
        print("Evaluate = {:.2f}".format(t2-t1))
        #import ipdb; ipdb.set_trace()
    
if __name__ == '__main__':
    main()        