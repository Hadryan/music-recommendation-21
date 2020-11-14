# Music-recommendation

**The architecture was inspired by [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)[![GitHub stars](https://img.shields.io/github/stars/hexiangnan/neural_collaborative_filtering.svg?logo=github&label=Stars)]**

---

## Overview

### Dataset
**Dataset : [Melon playlist](https://arena.kakao.com/c/8) is used** 

  - [x] **Users : 105141** 
  - [x] **Songs : 35919**  

## Files
> `split.py`: prepare train/test dataset
>
> `Data_Loader.py`: preprocess
>
> `metrics.py`: evaluation metrics including hit ratio(HR) and NDCG
>
> `model.py`: NCF model
>
> `evaluate.py`: evaluate
>
> `main.py`: main file

---

### Model 

<img width='768' src='https://user-images.githubusercontent.com/52492949/98954675-60ad3c80-2541-11eb-9b62-723bccbcf860.png'>

### Metric 

- [x] **NDCG@10**
- [x] **HR@10** 

---

## How to use 

### Languages 

<p align="left">
  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/languages/python.svg" alt="python" style="vertical-align:top; margin:6px 4px">
  </a> 

</p>

### Tools

<p align="left">
  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/tools/docker.svg" alt="docker" style="vertical-align:top; margin:6px 4px">
  </a> 

  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/tools/bash.svg" alt="bash" style="vertical-align:top; margin:6px 4px">
  </a> 

  <a href="#">
    <img src="https://github.com/MikeCodesDotNET/ColoredBadges/blob/master/svg/dev/tools/visualstudio_code.svg" alt="visualstudio_code" style="vertical-align:top; margin:6px 4px">
  </a> 

</p>

---

## Experiment 


<details><summary><b>Click me :heavy_exclamation_mark:<b></summary>

<details>
    <summary>  <b>Negative Samples : 1,5,10<b> 
    </summary>
<div markdown="1">


| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| <b>0.7502<b>|   0.4697|      1     |      4     |     1     |
| 0.7328|   <b>0.4705<b>|      5     |      4     |     1     |
| 0.6362|   0.4021|      10    |      4     |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7912|   0.5140|      1     |      8     |     1     |
| <b>0.8013<b>|   <b>0.5444<b>|      5     |      8     |     1     |
| 0.7469|   0.5026|      10    |      8     |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| <b>0.8224<b>|   0.5610|      1     |      16    |     1     |
| 0.8193|   <b>0.5795<b>|      5     |      16    |     1     |
| 0.7984|   0.5598|      10    |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7678|   0.4896|      1     |      4     |     2     |
| <b>0.7757<b>|   <b>0.5152<b>|      5     |      4     |     2     |
| 0.7064|   0.4631|      10    |      4     |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7965|   0.5266|      1     |      8     |     2     |
| <b>0.8000<b>|   <b>0.5527<b>|      5     |      8     |     2     |
| 0.7481|   0.5055|      10    |      8     |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8152|   0.5576|      1     |      16    |     2     |
| <b>0.8193<b>|   <b>0.5795<b>|      5     |      16    |     2     |
| 0.7898|   0.5530|      10    |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7824|   0.5097|      1     |      4     |     3     |
| <b>0.7882<b>|   <b>0.5372<b>|      5     |      4     |     3     |
| 0.7185|   0.4769|      10    |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| <b>0.8030<b>|   0.5412|      1     |      8     |     3     |
| 0.8026|   <b>0.5524<b>|      5     |      8     |     3     |
| 0.7696|   0.5324|      10    |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| <b>0.8155<b>|   0.5590|      1     |      16    |     3     |
| 0.8152|   <b>0.5732<b>|      5     |      16    |     3     |
| 0.7860|   0.5465|      10    |      16    |     3     |

</div>
</details>


<details>
    <summary>  Embedding size : 4,8,16
    </summary>
<div markdown="1">

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7502|   0.4697|      1     |      4     |     1     |
| 0.7912|   0.5140|      1     |      8     |     1     |
| <b>0.8224<b>|   <b>0.5610<b>|      1     |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7328|   0.4705|      5     |      4     |     1     |
| 0.8013|   0.5444|      5     |      8     |     1     |
| <b>0.8193<b>|   <b>0.5795<b>|      5     |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.6362|   0.4021|      10    |      4     |     1     |
| 0.7469|   0.5026|      10    |      8     |     1     |
| <b>0.7984<b>|   <b>0.5598<b>|      10    |      16    |     1     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7678|   0.4896|      1     |      4     |     2     |
| 0.7965|   0.5266|      1     |      8     |     2     |
| <b>0.8152<b>|   <b>0.5576<b>|      1     |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7757|   0.5152|      5     |      4     |     2     |
| 0.8000|   0.5527|      5     |      8     |     2     |
| <b>0.8193<b>|   <b>0.5795<b>|      5     |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7064|   0.4631|      10    |      4     |     2     |
| 0.7481|   0.5055|      10    |      8     |     2     |
| <b>0.7898<b>|   <b>0.5530<b>|      10    |      16    |     2     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7824|   0.5097|      1     |      4     |     3     |
| 0.8030|   0.5412|      1     |      8     |     3     |
| <b>0.8155<b>|   <b>0.5590<b>|      1     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7882|   0.5372|      5     |      4     |     3     |
| 0.8026|   0.5524|      5     |      8     |     3     |
| <b>0.8152<b>|   <b>0.5732<b>|      5     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7185|   0.4769|      10    |      4     |     3     |
| 0.7696|   0.5324|      10    |      8     |     3     |
| <b>0.7860<b>|   <b>0.5465<b>|      10    |      16    |     3     |

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
| <b>0.7824<b>|   <b>0.5097<b>|      1     |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7328|   0.4705|      5     |      4     |     1     |
| 0.7757|   0.5152|      5     |      4     |     2     |
| <b>0.7882<b>|   <b>0.5372<b>|      5     |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.6362|   0.4021|      10    |      4     |     1     |
| 0.7064|   0.4631|      10    |      4     |     2     |
| <b>0.7185<b>|   <b>0.4769<b>|      10    |      4     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7912|   0.5140|      1     |      8     |     1     |
| 0.7965|   0.5266|      1     |      8     |     2     |
| <b>0.8030<b>|   <b>0.5412<b>|      1     |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8013|   0.5444|      5     |      8     |     1     |
| 0.8000|   <b>0.5527<b>|      5     |      8     |     2     |
| <b>0.8026<b>|   0.5524|      5     |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.7469|   0.5026|      10    |      8     |     1     |
| 0.7481|   0.5055|      10    |      8     |     2     |
| <b>0.7696<b>|   <b>0.5324<b>|      10    |      8     |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| <b>0.8224<b>|   <b>0.5610<b>|      1     |      16    |     1     |
| 0.8152|   0.5576|      1     |      16    |     2     |
| 0.8155|   0.5590|      1     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| 0.8193|   0.5795|      5     |      16    |     1     |
| <b>0.8193<b>|   <b>0.5795<b>|      5     |      16    |     2     |
| 0.8152|   0.5732|      5     |      16    |     3     |

| HR@10 | NDCG@10 | Num of Neg | Num Factor | Num Layer |
|:-----:|:-------:|:----------:|:----------:|:---------:|
| <b>0.7984<b>|   <b>0.5598<b>|      10    |      16    |     1     |
| 0.7898|   0.5530|      10    |      16    |     2     |
| 0.7860|   0.5465|      10    |      16    |     3     |

</div>
</details>


---

### Run Example 
```sh
python3 main.py --optim=adam --lr=1e-3 --epochs=20 --batch_size=1024 --latent_dim_mf=8 --num_layers=3 --num_neg=5 --l2=0.0 --gpu=2,3
``` 