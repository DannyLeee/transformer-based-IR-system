#!/usr/bin/env python
# coding: utf-8

# ## Initial

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# In[2]:


from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(str(dt2)[:-13] + '\t' + msg)


# In[3]:


LM = "bert-base-uncased"


# In[4]:


doc_df = pd.read_csv("./dataset/documents.csv")
doc_df = doc_df.set_index('doc_id')
doc_df = doc_df.fillna("")
doc_dict = doc_df.to_dict()['doc_text']
train_q_df = pd.read_csv("./dataset/train_queries.csv")
test_q_df = pd.read_csv("./dataset/test_queries.csv")


# In[5]:


doc_df.head()


# In[6]:


train_q_df.head()


# In[7]:


test_q_df.head()


# ## Preprocess

# In[27]:


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


# ### training

# In[ ]:


train_bert_data = df_2_bert("train", train_q_df)


# In[ ]:


torch.save(train_bert_data, "./dataset/bert_data.pt")


# In[8]:


train_bert_data = torch.load("./dataset/1+3_bert_data.pt")


# ### testing

# In[28]:


test_bert_data = df_2_bert("test", test_q_df)


# In[29]:


torch.save(test_bert_data, "./dataset/test_bert_data.pt")


# In[ ]:


test_bert_data = torch.load("./dataset/test_bert_data.pt")


# ### Dataset Class

# In[9]:


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class QD_PairDataset(Dataset):
    def __init__(self, mode, list_of_bert):
        assert mode in ["train", "test", "dev"]
        self.mode = mode
        self.list_of_bert = list_of_bert

    def __getitem__(self, idx):
        bert_dict = self.list_of_bert[idx] #(batch=4; 1 pos 3 neg)
        inputid = torch.tensor(bert_dict['input_ids'])
        tokentype = torch.tensor(bert_dict['token_type_ids'])
        attentionmask = torch.tensor(bert_dict['attention_mask'])
        q_id = bert_dict['q_id']
        doc_id = bert_dict['doc_id']
        if self.mode == "train":
            label = torch.tensor(bert_dict['label'])
            return inputid, tokentype, attentionmask, label, q_id, doc_id
        else:
            return inputid, tokentype, attentionmask, q_id, doc_id

    def __len__(self):
        return len(self.list_of_bert)


# ## Model Building

# In[10]:


from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(LM, return_dict=True)


# ## Model Training

# In[11]:


""" model setting (training)"""
from transformers import BertConfig, AdamW
BATCH_SIZE = 2
trainSet = QD_PairDataset("train", train_bert_data)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
optimizer = AdamW(model.parameters(), lr=1e-5) # AdamW = BertAdam
loss_fct = nn.CrossEntropyLoss()
weight = torch.FloatTensor([3, 1]).cuda()
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


# In[13]:


""" training """
model = model.to(device)
model.train()

EPOCHS = 5
timestamp("start training")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in tqdm(trainLoader):
        # print(data)
        # break
        tokens_tensors, segments_tensors, masks_tensors,         labels, q_id, doc_id = [t for t in data]

        tokens_tensors = tokens_tensors.to(device)
        segments_tensors = segments_tensors.to(device)
        masks_tensors = masks_tensors.to(device)
        labels = labels.to(device)

        tokens_tensors = torch.reshape(tokens_tensors, (8, 512))
        segments_tensors = torch.reshape(segments_tensors, (8, 512))
        masks_tensors = torch.reshape(masks_tensors, (8, 512))
        labels = torch.reshape(labels, (8,))

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
    CHECKPOINT_NAME = './model/bert_base_uncase_E_' + str(epoch+1) + '.pt'
    torch.save(model.state_dict(), CHECKPOINT_NAME)
    timestamp(f"[epoch {epoch+1}] loss: {running_loss:.3f}")


# ## Model Inference

# In[64]:


def get_predictions(model, testLoader, BATCH_SIZE):
    result = []
    with torch.no_grad():
        for data in tqdm(testLoader):
            data = [t for t in data if t is not None]

            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            tokens_tensors = tokens_tensors.to("cuda:0")
            segments_tensors = segments_tensors.to("cuda:0")
            masks_tensors = masks_tensors.to("cuda:0")

            outputs = model(input_ids=tokens_tensors, 
                      token_type_ids=segments_tensors, 
                      attention_mask=masks_tensors)

            softmax = nn.Softmax(dim=1)
            score = softmax(outputs.logits)[:, 1] # softmax and get 1 as score
            doc_id = data[4]

            for _, q_id in enumerate(data[3]):
                data_dict = {"q_id":q_id.item(), "doc_id":doc_id[_], "score":score[_].item()}
                result += [data_dict]
        
    return result


# In[65]:


"""testing"""
MODEL_PATH = "./model/bert_base_uncase_E_5.pt"
model.load_state_dict(torch.load(MODEL_PATH))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

BATCH_SIZE = 100
testSet = QD_PairDataset("test", test_bert_data)
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE)

predictions = get_predictions(model, testLoader, BATCH_SIZE)


# ## Output

# In[85]:


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


# ### BM_result

# In[84]:


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

