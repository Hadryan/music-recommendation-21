## Kakao Music recommendation : A Pytorch Implementation 

**The architecture was inspired by [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)]**

---

## Requirements 

```shell
Cuda 11.0
Python3 3.8
PyTorch 1.8 
Torchvision 0.10.0
```

## Quickstart

### Weights & Biases(Visualization tool)

- Before starting, you should login wandb using your personal API key. 
- Weights & Biases : https://wandb.ai/site

```shell
!pip install wandb
wandb login PERSONAL_API_KEY
```

### Cloning a repository

```shell
git clone https://github.com/ssuncheol/music-recommendation.git
```

### Prepare a dataset

- Download [Melon playlist](https://arena.kakao.com/c/8) from open-source 

  -  Users : 105141 
  -  Songs : 35919 

### Files
> `split.py`: prepare train/test dataset
>
> `dataloader.py`: preprocess
>
> `metrics.py`: evaluation metrics including hit ratio(HR) and NDCG
>
> `model.py`: NCF model
>
> `evaluate.py`: evaluate
>
> `main.py`: main file

---

## Model 

I use neural collaborative filtering. model's structure as follows 

<img width='768' src='https://user-images.githubusercontent.com/52492949/98954675-60ad3c80-2541-11eb-9b62-723bccbcf860.png'>

### Evaluation metrics

- NDCG@10
- HR@10

---

## Experiments 

I use Melon-playlist Dataset to train and evaluate model 

### Arguments
| Args 	| Type 	| Description 	| Default|
|:---------:|:--------:|:----------------------------------------------------:|:-----:|
| Epochs 	| [int] 	| Epochs | 20|
| Batch_size 	| [int] 	| Batch size| 1024|
| Optimizer 	| [str]	| Adam| 	Adam|
| Batch size 	| [int]	| Batch size| 	1024|
| Latent dim 	| [int]	| Latent dim| 	8|
| Num layers 	| [int]	| Num layers| 	3|
| Negative samples 	| [int]	| Negative samples| 	4|
| Learning rate 	| [float]	| Learning rate | 1e-3|
| L2-norm	| [float]	| L2-norm | 0.0|
| GPU 	| [str]	| GPU | '0' |


### How to train 

```sh
python3 main.py --optim=adam --lr=1e-3 --epochs=20 --batch_size=1024 --latent_dim_mf=8 --num_layers=3 --num_neg=4 --l2=0.0 --gpu=2,3
``` 



---

## Experiment results 

<details>
    <summary>  <b> Negative Samples : 1,5,10<b> 
    </summary>
<div markdown="1">

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7502|   0.4697|      1     |      4     |     1     |
| 0.7328|   0.4705|      5     |      4     |     1     |
| 0.6362|   0.4021|      10    |      4     |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7912|   0.5140|      1     |      8     |     1     |
| 0.8013|   0.5444|      5     |      8     |     1     |
| 0.7469|   0.5026|      10    |      8     |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8224|   0.5610|      1     |      16    |     1     |
| 0.8193|   0.5795|      5     |      16    |     1     |
| 0.7984|   0.5598|      10    |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7678|   0.4896|      1     |      4     |     2     |
| 0.7757|   0.5152|      5     |      4     |     2     |
| 0.7064|   0.4631|      10    |      4     |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7965|   0.5266|      1     |      8     |     2     |
| 0.8000|   0.5527|      5     |      8     |     2     |
| 0.7481|   0.5055|      10    |      8     |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8152|   0.5576|      1     |      16    |     2     |
| 0.8193|   0.5795|      5     |      16    |     2     |
| 0.7898|   0.5530|      10    |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7824|   0.5097|      1     |      4     |     3     |
| 0.7882|   0.5372|      5     |      4     |     3     |
| 0.7185|   0.4769|      10    |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8030|   0.5412|      1     |      8     |     3     |
| 0.8026|   0.5524|      5     |      8     |     3     |
| 0.7696|   0.5324|      10    |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8155|   0.5590|      1     |      16    |     3     |
| 0.8152|   0.5732|      5     |      16    |     3     |
| 0.7860|   0.5465|      10    |      16    |     3     |

</div>
</details>


<details>
    <summary>  Num Factor : 4,8,16
    </summary>
<div markdown="1">

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7502|   0.4697|      1     |      4     |     1     |
| 0.7912|   0.5140|      1     |      8     |     1     |
| 0.8224|   0.5610|      1     |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7328|   0.4705|      5     |      4     |     1     |
| 0.8013|   0.5444|      5     |      8     |     1     |
| 0.8193|   0.5795|      5     |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.6362|   0.4021|      10    |      4     |     1     |
| 0.7469|   0.5026|      10    |      8     |     1     |
| 0.7984|   0.5598|      10    |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7678|   0.4896|      1     |      4     |     2     |
| 0.7965|   0.5266|      1     |      8     |     2     |
| 0.8152|   0.5576|      1     |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7757|   0.5152|      5     |      4     |     2     |
| 0.8000|   0.5527|      5     |      8     |     2     |
| 0.8193|   0.5795|      5     |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7064|   0.4631|      10    |      4     |     2     |
| 0.7481|   0.5055|      10    |      8     |     2     |
| 0.7898|   0.5530|      10    |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7824|   0.5097|      1     |      4     |     3     |
| 0.8030|   0.5412|      1     |      8     |     3     |
| 0.8155|   0.5590|      1     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7882|   0.5372|      5     |      4     |     3     |
| 0.8026|   0.5524|      5     |      8     |     3     |
| 0.8152|   0.5732|      5     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7185|   0.4769|      10    |      4     |     3     |
| 0.7696|   0.5324|      10    |      8     |     3     |
| 0.7860|   0.5465|      10    |      16    |     3     |

</div>
</details>

<details>
    <summary>  Num Layer : 1,2,3
    </summary>
<div markdown="1">

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7502|   0.4697|      1     |      4     |     1     |
| 0.7678|   0.4896|      1     |      4     |     2     |
| 0.7824|   0.5097|      1     |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7328|   0.4705|      5     |      4     |     1     |
| 0.7757|   0.5152|      5     |      4     |     2     |
| 0.7882|   0.5372|      5     |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.6362|   0.4021|      10    |      4     |     1     |
| 0.7064|   0.4631|      10    |      4     |     2     |
| 0.7185|   0.4769|      10    |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7912|   0.5140|      1     |      8     |     1     |
| 0.7965|   0.5266|      1     |      8     |     2     |
| 0.8030|   0.5412|      1     |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8013|   0.5444|      5     |      8     |     1     |
| 0.8000|   0.5527|      5     |      8     |     2     |
| 0.8026|   0.5524|      5     |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7469|   0.5026|      10    |      8     |     1     |
| 0.7481|   0.5055|      10    |      8     |     2     |
| 0.7696|   0.5324|      10    |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8224|   0.5610|      1     |      16    |     1     |
| 0.8152|   0.5576|      1     |      16    |     2     |
| 0.8155|   0.5590|      1     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8193|   0.5795|      5     |      16    |     1     |
| 0.8193|   0.5795|      5     |      16    |     2     |
| 0.8152|   0.5732|      5     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7984|   0.5598|      10    |      16    |     1     |
| 0.7898|   0.5530|      10    |      16    |     2     |
| 0.7860|   0.5465|      10    |      16    |     3     |

</div>
</details>


---

## Reference 
He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
