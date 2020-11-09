import pandas as pd
import numpy as np
import warnings
import random
from random import sample 
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")

train = pd.read_json("/daintlab/data/music_rec/train.json")
song_meta = pd.read_json("/daintlab/data/music_rec/song_meta.json")
filtering = 25

def to_date(x):
    try:
        y = pd.to_datetime(x, format = "%Y%m%d")
    except:
        y = None
    return y

train_song = train['songs']
song_counter = Counter([song for songs in train_song for song in songs])
song_dict = {x: song_counter[x] for x in song_counter}
song_dict = dict(filter(lambda x : x[1]>=filtering, song_dict.items())) # filtering song

song_id_sid = dict()
for i, song_id in enumerate(song_dict): 
    song_id_sid[song_id] = i 


song_meta = song_meta[song_meta["id"].notnull()]
song_meta["id"] = song_meta["id"].astype(int)
song_meta["issue_date"] = song_meta["issue_date"].astype(str)
song_meta["issue_date"] = song_meta["issue_date"].apply(lambda x : x[:4] + x[4:6].replace("00","01") + x[6:].replace("00","01"))
song_meta["issue_date"] = song_meta["issue_date"].apply(lambda x : to_date(x))
song_meta = song_meta[song_meta["issue_date"].notnull()] # 결측치 제거
song_meta["timestamp"] = song_meta["issue_date"].apply(lambda x : datetime.timestamp(x))
song_meta["itemId"] = song_meta["id"].apply(lambda x : song_id_sid.get(x))
song_meta = song_meta[song_meta["itemId"].notnull()]

issue_date = dict(zip(song_meta["itemId"],song_meta["timestamp"]))

items = set(issue_date.keys())
print(max(song_id_sid.values()))
num_items = max(issue_date.keys()) + 1
print("n_items = {}".format(num_items))

train['itemId'] = train['songs'].apply(lambda x: [song_id_sid.get(item) for item in x if song_id_sid.get(item) != None])
train['itemId'] = train['itemId'].apply(lambda x: [item for item in x if issue_date.get(item) != None])
train.loc[:,'num_items'] = train['itemId'].map(len)
train = train[train["num_items"]>1]
n_data = len(train)
train["userId"] = range(n_data)

train["timestamp"] = train["itemId"].apply(lambda x: [issue_date.get(item) for item in x if issue_date.get(item) != None])
train["test_index"] = train["timestamp"].apply(lambda x : np.array(x).argmax())
train["test_rating"] = train.apply(lambda x: x["itemId"][x["test_index"]], axis = 1)
train["test"] = train["itemId"].apply(lambda x : list(items - set(x)))
train["test_negative"] = train["test"].apply(lambda x : random.sample(x,99))
train["train_negative"] = train.apply(lambda x : list(items - set(x["itemId"]) - set(x["test_negative"])), axis = 1)
train.apply(lambda x : x["itemId"].remove(x["test_rating"]), axis = 1)
train.rename(columns = {"itemId":"train_positive"}, inplace = True)
train = train[["userId","train_positive","train_negative","test_rating","test_negative"]].reset_index()
print(train)

#https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
train.to_feather("melon_"+str(num_items)+".ftr")