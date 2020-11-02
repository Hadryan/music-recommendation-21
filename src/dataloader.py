import torch
import random
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset 
from torch.utils.data.dataloader import Sampler

#seed 
random.seed(42)

class UserItemRatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

#shuffle data
    
class MySampler(Sampler):
    def __init__(self,data_source):
        
        self.seq = list(range(len(data_source)))
    def __iter__(self): 
        return iter(self.seq)
    
    
class SampleGenerator(object):
    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        assert 'userId' in ratings.columns
        assert 'itemId' in ratings.columns
        assert 'rating' in ratings.columns

        self.ratings = ratings
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.negatives = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.ratings)
    
    def _sample_negative(self, ratings):
        #make 100 negative sampling
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x,99))
        return interact_status[['userId', 'negative_items','negative_samples']]


    def _split_loo(self, ratings):
        # leave one out 
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        print("Split Complete")
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]
    


    def instance_a_train_loader(self, batch_size):
        user = np.array(self.ratings["userId"])
        item = np.array(self.ratings["itemId"])
        rating = np.array(self.ratings["rating"])        
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user),
                                        item_tensor=torch.LongTensor(item),
                                        target_tensor=torch.FloatTensor(rating))
        sampler = MySampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=0,sampler=sampler)


    @property
    def evaluate_data(self):
        #maek evaluate data
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        print("test dataset complete")
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
