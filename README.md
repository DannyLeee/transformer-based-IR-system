## Introduction
* This is the Information Retrieval HW6.
* Using BM25 and transformer-based language model to compute the relation between given querys and documents.

## Usage
* All file have argument parser, please use `python3 FILENAME -h` to get more.

## Approach
* Use transformer-based doing multiple choice task.

### preprocess
* `[CLS]` query `[SEP]` document `[SEP]`
* Token length truncate to 512 by tokenizer.

### Hyperparameter
* Pretrained lenguage model: bert-base-uncased
* Optimizer: AdamW 
* Learning rate = 1e-5
* Split 20 documents of training queries to grid search optimal 𝜶 for BERT
* Batch size = 3
* Num. epochs = 2
* Num. of negative documents = 3
* 𝜶 from greedy search = 1.13

<!-- <img src="https://latex.codecogs.com/gif.latex?[formula]"/> -->