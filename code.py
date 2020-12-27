"""##Initial"""

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(str(dt2)[:-13] + '\t' + msg)

LM = "bert-base-uncased"

doc_df = pd.read_csv("./dataset/documents.csv")
doc_df = doc_df.set_index('doc_id')
doc_df = doc_df.fillna("")
doc_dict = doc_df.to_dict()['doc_text']
train_q_df = pd.read_csv("./dataset/train_queries.csv")
test_q_df = pd.read_csv("./dataset/test_queries.csv")

doc_df.head()

train_q_df.head()

test_q_df.head()

"""## Preprocess"""

import random
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(LM)

def df_2_bert(mode, df):
    assert mode in ["train", "test", "dev"]
    bert_data = []
    for index, row in tqdm(df.iterrows()):
        q_id = row.query_id
        query = row.query_text
        
        if mode == "train":
            # 1 positive, 3 negative
            neg_doc = list(set(row.bm25_top1000.split()) - set(row.pos_doc_ids.split()))
            for r_doc in row.pos_doc_ids.split():
                bert_dict = tokenizer(query, doc_dict[r_doc],
                                        max_length=512,
                                        padding='max_length',
                                        truncation=True) # dict of tensor {ids:[]...}
                bert_dict['q_id'] = q_id
                bert_dict['doc_id'] = r_doc
                bert_dict['label'] = 1
                bert_data += [bert_dict]
                sampled_neg_doc = random.sample(neg_doc, 3) # 3 negative
                for nr_doc in sampled_neg_doc:
                    bert_dict = tokenizer(query, doc_dict[nr_doc],
                                        max_length=512,
                                        padding='max_length',
                                        truncation=True) # dict of tensor {ids:[]...}
                    bert_dict['q_id'] = q_id
                    bert_dict['doc_id'] = nr_doc
                    bert_dict['label'] = 0
                    bert_data += [bert_dict]
        elif mode  == "test":
            for doc in row.bm25_top1000.split():
                bert_dict = tokenizer(query, doc_dict[doc],
                                        max_length=512,
                                        padding='max_length',
                                        truncation=True) # dict of tensor {ids:[]...}
                bert_dict['q_id'] = q_id
                bert_dict['doc_id'] = doc
                bert_data += [bert_dict]
            
    return bert_data # List[Dict[List]] = List[tokenizer output]

"""### training"""

train_bert_data = df_2_bert("train", train_q_df)

torch.save(train_bert_data, "./dataset/full_bert_data.pt")

train_bert_data = torch.load("./dataset/1+3_bert_data.pt")

"""### testing"""

test_bert_data = df_2_bert("test", test_q_df)

torch.save(test_bert_data, "./dataset/test_bert_data.pt")

test_bert_data = torch.load("./dataset/test_bert_data.pt")

"""### Dataset Class"""

import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class QD_PairDataset(Dataset):
    def __init__(self, mode, list_of_bert):
        assert mode in ["train", "test", "dev"]
        self.mode = mode
        self.list_of_bert = list_of_bert

    def __getitem__(self, idx):
        bert_dict = self.list_of_bert[idx] #(batch=4; 1 pos 3 neg)
        if self.mode == "train":
            #shuffle
            rand_idx = list(range(4))
            random.shuffle(rand_idx)
            s_inputid = []
            s_tokentype = []
            s_att = []
            s_label = []
            s_q_id = []
            s_doc_id = []
            for i in rand_idx:
                s_inputid += [bert_dict['input_ids'][i]]
                s_tokentype += [bert_dict['token_type_ids'][i]]
                s_att += [bert_dict['attention_mask'][i]]
                s_label += [bert_dict['label'][i]]
                s_q_id += [bert_dict['q_id'][i]]
                s_doc_id += [bert_dict['doc_id'][i]]
            inputid = torch.tensor(s_inputid)
            tokentype = torch.tensor(s_tokentype)
            attentionmask = torch.tensor(s_att)
            label = torch.tensor(s_label.index(1)) # multiple choise label

            return inputid, tokentype, attentionmask, label, s_q_id, s_doc_id
        else:
            inputid = torch.tensor([bert_dict['input_ids']])
            tokentype = torch.tensor([bert_dict['token_type_ids']])
            attentionmask = torch.tensor([bert_dict['attention_mask']])
            q_id = [bert_dict['q_id']]
            doc_id = [bert_dict['doc_id']]

            return inputid, tokentype, attentionmask, q_id, doc_id

    def __len__(self):
        return len(self.list_of_bert)

"""## Model Building"""

from transformers import BertForMultipleChoice
model = BertForMultipleChoice.from_pretrained(LM, return_dict=True)

"""##Model Training"""

""" model setting (training)"""
from transformers import BertConfig, AdamW
BATCH_SIZE = 3
trainSet = QD_PairDataset("train", train_bert_data)##########
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
optimizer = AdamW(model.parameters(), lr=1e-5) # AdamW = BertAdam
# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
#             print(_)
    else:
        print("{:15} {}".format(name, module))

""" training """
MODEL_PATH = "./model/mul_bert_base_uncase_E_5.pt"
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.train()
train_from = 5
EPOCHS = 15
timestamp("start training")
for epoch in range(train_from, EPOCHS):
    running_loss = 0.0
    for data in tqdm(trainLoader):
        # print(data)
        # break
        tokens_tensors, segments_tensors, masks_tensors, \
        labels, q_id, doc_id = [t for t in data]

        tokens_tensors = tokens_tensors.to(device)
        segments_tensors = segments_tensors.to(device)
        masks_tensors = masks_tensors.to(device)
        labels = labels.to(device)

        # 將參數梯度歸零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids = tokens_tensors, 
                        token_type_ids = segments_tensors, 
                        attention_mask = masks_tensors,
                        labels = labels)
        
        loss = outputs.loss
        # backward
        loss.backward()
        optimizer.step()

        # 紀錄當前 batch loss
        running_loss += loss.item()
    #     break
    # break
    CHECKPOINT_NAME = './model/mul_bert_base_uncase_E_' + str(epoch+1) + '.pt'
    torch.save(model.state_dict(), CHECKPOINT_NAME)
    timestamp(f"[epoch {epoch+1}] loss: {running_loss:.3f}")

"""## Model Inference"""

def get_predictions(model, testLoader, BATCH_SIZE):
    result = []
    with torch.no_grad():
        for data in tqdm(testLoader):
            data = [t for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            # tokens_tensors = tokens_tensors.to("cuda:0")
            # segments_tensors = segments_tensors.to("cuda:0")
            # masks_tensors = masks_tensors.to("cuda:0")

            # print(tokens_tensors.shape)
            # break

            outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

            # softmax = nn.Softmax(dim=1)
            # score = softmax(outputs.logits)[:, 1] # softmax and get 1 as score
            
            score = outputs.logits.view(-1)
            
            doc_id = data[4]

            for _, q_id in enumerate(data[3]):
                data_dict = {"q_id":q_id.item(), "doc_id":doc_id[_], "score":score[_].item()}
                result += [data_dict]
        
    return result

"""testing"""
MODEL_PATH = "./model/mul_bert_base_uncase_E_11.pt"
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))) ###

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BATCH_SIZE = 100
testSet = QD_PairDataset("test", test_bert_data)
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE)

predictions = get_predictions(model, testLoader, BATCH_SIZE)

"""## Output"""

import numpy as np
test_q_list = test_q_df['query_id']
test_doc_list = test_q_df['bm25_top1000']
test_doc_score = test_q_df['bm25_top1000_scores']

A = 1
with open('result.csv', 'w') as fp:
    fp.write("query_id,ranked_doc_ids\n")
    for i, q_id in tqdm(enumerate(test_q_list)):
        fp.write(str(q_id)+',')
        bm_score = np.array([float(s) for s in test_doc_score[i].split()])
        bert_score = []
        for j in range(1000):
            bert_score += [predictions[i+j]['score']]
        bert_score = np.array(bert_score)
        score = bm_score + A*bert_score
        sortidx = np.argsort(score)
        sortidx = np.flip(sortidx)
        doc_list = test_doc_list[i].split()
        for idx in sortidx:
            fp.write(doc_list[idx]+' ')
        fp.write("\n")
timestamp("output done")

"""### BM_result"""

import numpy as np
test_q_list = test_q_df['query_id']
test_doc_list = test_q_df['bm25_top1000']
test_doc_score = test_q_df['bm25_top1000_scores']

A = 1
with open('bm_result.csv', 'w') as fp:
    fp.write("query_id,ranked_doc_ids\n")
    for i, q_id in tqdm(enumerate(test_q_list)):
        fp.write(str(q_id) + ',' + test_doc_list[i] + ' ')
        fp.write("\n")

"""## Validate"""

val_df = train_q_df[:30]
val_bert_data = df_2_bert("test", val_df)

torch.save(val_bert_data, "./dataset/val_df.pt")

MODEL_PATH = "./model/bert_base_uncase_E_5.pt"
model.load_state_dict(torch.load(MODEL_PATH))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BATCH_SIZE = 100
valSet = QD_PairDataset("test", val_bert_data)
valLoader = DataLoader(valSet, batch_size=BATCH_SIZE)

predictions = get_predictions(model, valLoader, BATCH_SIZE)

import numpy as np
val_q_list = val_df['query_id']
val_doc_list = val_df['bm25_top1000']
val_doc_score = val_df['bm25_top1000_scores']

A = 1
with open('val_result.csv', 'w') as fp:
    fp.write("query_id,ranked_doc_ids\n")
    for i, q_id in tqdm(enumerate(val_q_list)):
        fp.write(str(q_id)+',')
        bm_score = np.array([float(s) for s in val_doc_score[i].split()])
        bert_score = []
        for j in range(1000):
            bert_score += [predictions[i+j]['score']]
        bert_score = np.array(bert_score)
        score = bm_score + A*bert_score
        sortidx = np.argsort(score)
        sortidx = np.flip(sortidx)
        doc_list = val_doc_list[i].split()
        for idx in sortidx:
            fp.write(doc_list[idx]+' ')
        fp.write("\n")
timestamp("output done")

