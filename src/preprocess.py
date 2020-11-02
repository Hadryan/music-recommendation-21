import warnings
import pandas as pd
import numpy as np
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
    song_dict = dict(filter(lambda x : x[1]>=10, song_dict.items())) # filtering song

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
    
    return data