# Music-recommendation

**The architecture was inspired by [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)]**

---

## Overview

### Dataset
**Dataset : [Melon playlist](https://arena.kakao.com/c/8) is used** 

- [x] **Users : 105141** 
- [x] **Songs : 35919**  

### Dataframe 

**Columns = ["userId","train_positive","train_negative","test_rating","test_negative"]**<br>
>
`userId` : user id<br>
>
`train_positive` : interacted song<br>
>
`test_rating` : Leave one out으로 고른 test song (곡 발매일 기준으로 최신 곡을 test song으로 분류)<br>
>
`test_negative` : train_positive와 test_rating에 없는 곡들 중 랜덤하게 뽑은 각 유저당 99개의 곡들의 집합<br>
>
`train_negative` : 각 interacted song당 train_positive, test_rating, test_negative와 겹치지 않는 모든 곡들의 집합<br> 

### Model 

<img width='768' src='https://user-images.githubusercontent.com/52492949/98676852-7edb3700-239f-11eb-91e3-e6f40c2ece45.png'>

### Metric 

- [x] **NDCG**
- [x] **HR** 

---

## How to use 

### Dependencies

- [x] **Pytorch** 


## Experiment results

- [x] **Num of Neg : 1,5,10**<br> 
>
- [x] **Num Factor : 8,16,32**<br> 
>
- [x] **Num Layer : 1,2,3**<br>

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7912|   0.5140|      1     |      8     |     1     |
| 0.7880|   0.5260|      5     |      8     |     1     |
| -     |  -      |      10    |      8     |     1     |
| -     |  -      |      1     |      16    |     1     |
| -     |  -      |      5     |      16    |     1     |
| -     |  -      |      10    |      16    |     1     |
| -     |  -      |      1     |      32    |     1     |
| -     |  -      |      5     |      32    |     1     |
| -     |  -      |      10    |      32    |     1     |
| 0.8030|   0.5412|      1     |      8     |     3     |
| 0.8026|   0.5524|      5     |      8     |     3     |
| 0.7696|   0.5324|      10    |      8     |     3     |
| -     |  -      |      1     |      16    |     3     |
| -     |  -      |      5     |      16    |     3     |
| -     |  -      |      10    |      16    |     3     |
| -     |  -      |      1     |      32    |     3     |
| -     |  -      |      5     |      32    |     3     |
| -     |  -      |      10    |      32    |     3     |


### Run Example 
```sh
python3 main.py --optim=adam --lr=1e-3 --epochs=20 --batch_size=1024 --latent_dim_mf=8 --num_layers=3 --num_neg=5 --l2=0.0 --gpu=2,3
``` 