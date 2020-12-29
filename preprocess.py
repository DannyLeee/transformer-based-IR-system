import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(str(dt2)[:-13] + '\t' + msg)

parser = argparse.ArgumentParser()
parser.add_argument("-LM",default="bert-base-uncased", type=str)
parser.add_argument("-bert_data_path", type=str, required=True) # "./dataset/1+3_bert_data.pt"
parser.add_argument("-mode", type=str, choices=["train", "test"], required=True)
args = parser.parse_args()

doc_df = pd.read_csv("./dataset/documents.csv")
doc_df = doc_df.set_index('doc_id')
doc_df = doc_df.fillna("")
doc_dict = doc_df.to_dict()['doc_text']
train_q_df = pd.read_csv("./dataset/train_queries.csv")
test_q_df = pd.read_csv("./dataset/test_queries.csv")

import random
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(args.LM)

def df_2_bert(mode, df, document_dict):
    assert mode in ["train", "test", "dev"]
    bert_data = []
    q_id_list = df['query_id']
    q_list = df['query_text']
    if mode == "train":
        pos_doc_ids_list = df['pos_doc_ids']
    bm25_top1000_list = df['bm25_top1000']
    all_used_doc = []
    for doc_list in bm25_top1000_list:
        all_used_doc += [doc for doc in doc_list.split()]

    for idx in range(len(q_id_list)):
        pos_doc_ids = pos_doc_ids_list[idx].split()
        for doc_list in pos_doc_ids:
            all_used_doc += [doc_list]
    all_used_doc = list(set(all_used_doc))
    doc_dict = {key: document_dict[key] for key in all_used_doc} 
    print(len(doc_dict))

    for idx, q_id in tqdm(enumerate(q_id_list)):
        query = q_list[idx]
        if mode == "train":
            # 1 positive, 3 negative
            neg_doc = list(set(bm25_top1000_list[idx].split()) - set(pos_doc_ids_list[idx].split()))
            
            for r_doc in pos_doc_ids_list[idx].split():
                batch_q = [query]*4
                batch_doc = [doc_dict[r_doc]]

                sampled_neg_doc = random.sample(neg_doc, 3) # 3 negative
                for nr_doc in sampled_neg_doc:
                        batch_doc += [doc_dict[nr_doc]]
                bert_dict = tokenizer(batch_q, batch_doc,
                                        max_length=512,
                                        padding='max_length',
                                        return_token_type_ids=True,
                                        truncation=True) # dict of tensor {ids:[]...}
                bert_dict['q_id'] = [q_id]*4
                bert_dict['doc_id'] = [r_doc] + sampled_neg_doc
                bert_dict['label'] = [1] + [0]*3
                bert_data += [bert_dict]

        elif mode  == "test":
            for doc in bm25_top1000_list[idx].split():
                bert_dict = tokenizer(query, doc_dict[doc],
                                        max_length=512,
                                        padding='max_length',
                                        return_token_type_ids=True,
                                        truncation=True) # dict of tensor {ids:[]...}
                bert_dict['q_id'] = q_id
                bert_dict['doc_id'] = doc
                bert_data += [bert_dict]
            
    return bert_data # List[Dict[List]] = List[tokenizer output]

train_bert_data = df_2_bert(args.mode, train_q_df, doc_dict)
torch.save(train_bert_data, args.bert_data_path)