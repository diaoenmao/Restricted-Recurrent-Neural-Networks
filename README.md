# Restricted Recurrent Neural Networks

[IEEE BigData 2019] This is an implementation of [Restricted Recurrent Neural Networks](https://arxiv.org/abs/1908.07724)
![illustration](/img/illustration.png)
 
## Requirements
 - Python 3
 - PyTorch 1.0

## Results
- Model Complexity of RRNN and its variants (unit: million)  

| r    |    1   |   0.95 | 0.9    | 0.7    | 0.5    | 0.3    | 0.1    | 0      |
|:------:|:------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| RRNN  | 0.130  | 0.136  | 0.142  | 0.167  | 0.191  | 0.215  | 0.239  | 0.251  |
| RGRU  | 0.130  | 0.161  | 0.191  | 0.311  | 0.432  | 0.553  | 0.673  | 0.733  |
| RLSTM | 0.130  | 0.173  | 0.215  | 0.384  | 0.553  | 0.721  | 0.890  | 0.975  |
- Comparison with state-of-the-art architectures in terms of Test Perplexity on Penn Treebank dataset 

| Model 	| Model parameters (M) 	| Test Perplexity 	|
|:------------------------:	|:--------------------:	|:----------------:	|
| LR LSTM 200-200 	| 0.928 	| 136.115 	|
| LSTM-SparseVD-VOC  	| 1.672 	| 120.2 	|
| KN5 + cache 	| 2 	| 125.7 	|
| LR LSTM 400-400 	| 3.28 	| 106.623 	|
| LSTM-SparseVD 	| 3.312 	| 109.2 	|
| RNN-LDA + KN-5 + cache 	| 9 	| 92 	|
| AWD-LSTM 	| 22 	| 55.97 	|
| RLSTM-Tied-Dropout (s=0.5) 	| 2.553 (0.553) 	| 103.5 	|
- Perplexity vs. Number of RNN parameters for Penn Treebank dataset.

![Penn Treebank](/img/PennTreebank.png)

- Perplexity vs. Number of RNN parameters for WikiText2 dataset.

![WikiText2](/img/WikiText2.png)

## Acknowledgement
*Enmao Diao  
Jie Ding  
Vahid Tarokh*
