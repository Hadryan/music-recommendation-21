import warnings
import time
import pandas as pd
import numpy as np
import random
from random import sample 
from collections import Counter


#to_datetime

def to_date(x):
    try:
        y = pd.to_datetime(x, format = "%Y%m%d")
    except:
        y = None
    return y

#preprocess

def pre(train,song_meta):
    warnings.filterwarnings("ignore")
    n_data = len(train)
    train["nid"] = range(n_data) 
    train_song = train['songs']
    song_counter = Counter([song for songs in train_song for song in songs])
    song_dict = {x: song_counter[x] for x in song_counter}
    song_dict = dict(filter(lambda x : x[1]>=300, song_dict.items())) # filtering song

    song_id_sid = dict()
    song_sid_id = dict()
    for i, song_id in enumerate(song_dict): 
        song_id_sid[song_id] = i 
        song_sid_id[i] = song_id 
    
    
    n_songs = len(song_dict)
    song_meta["nid"] = song_meta["id"].apply(lambda x : song_id_sid.get(x))
    meta = song_meta[song_meta["nid"].notnull()]
    meta["nid"] = meta["nid"].astype(int)
    meta["issue_date"] = meta["issue_date"].astype(str)
    meta["issue_date"] = meta["issue_date"].apply(lambda x : x[:4] + x[4:6].replace("00","01") + x[6:].replace("00","01"))
    meta["issue_date"] = meta["issue_date"].apply(lambda x : to_date(x))
    meta = meta[meta["issue_date"].notnull()]

    issue_date = dict(zip(meta["nid"],meta["issue_date"]))

    train['itemId'] = train['songs'].map(lambda x: [song_id_sid.get(song_id) for song_id in x if song_id_sid.get(song_id) != None])
    train.loc[:,'num_songs'] = train['itemId'].map(len)

    user_train_1 = np.repeat(range(n_data), train['num_songs'])
    song_train_1 = np.array([song for songs in train['itemId'] for song in songs])
    rat_train_1 = np.repeat(1, train['num_songs'].sum())
    dat_train_1 = np.array([issue_date.get(song) for songs in train['itemId'] for song in songs])
    data = pd.DataFrame({"userId":user_train_1,"itemId":song_train_1,"rating":rat_train_1,"timestamp":dat_train_1})

    data['rank_latest'] = data.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    cand = []
    t1 = time.time()
    for songs in train['itemId']:
        tot_songs = np.arange(n_songs)
        #np.random.seed(42)
        cand_0 = sample(tot_songs[np.isin(tot_songs,songs) == False].tolist(),4)
        for c0 in cand_0:
            cand.append(c0)
    t2 = time.time()
    user_train_0 = np.repeat(range(n_data), 4).reshape(-1,1)
    song_train_0 = np.array(cand).reshape(-1,1)
    rate_train_0 = np.repeat(0, n_data*4).reshape(-1,1)
    print("Negative Sampling Time = ", t2-t1)
    inputs = np.hstack([user_train_0,song_train_0,rate_train_0])
    inputs = pd.DataFrame(inputs, columns = ["userId","itemId","rating"])

    data = pd.concat([data, inputs])
    return data,n_data,n_songs