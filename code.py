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
train_q_df = pd.read_csv("./dataset/train_queries.csv")
test_q_df = pd.read_csv("./dataset/test_queries.csv")


# In[5]:


doc_df.head()


# In[6]:


train_q_df.head()


# In[7]:


test_q_df.head()


# ## Preprocess

# In[15]:


import random
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(LM)

def df_2_bert(mode, df):
    assert mode in ["train", "test", "dev"]
    bert_data = []
    for index, row in tqdm(df.iterrows()):
        q_id = row.query_id
        query = row.query_text
        
        # 1 positive, 3 negative
        neg_doc = list(set(row.bm25_top1000.split()) - set(row.pos_doc_ids.split()))
        for r_doc in row.pos_doc_ids.split():
            bert_dict = tokenizer(query, doc_df.loc[r_doc, 'doc_text'],
                                    max_length=512,
                                    padding='max_length',
                                    truncation=True) # dict of tensor {ids:[]...}
            bert_dict['q_id'] = q_id
            bert_dict['doc_id'] = r_doc
            bert_dict['label'] = 1
            bert_data += [bert_dict]
            sampled_neg_doc = random.sample(neg_doc, 3) # 3 negative
            for nr_doc in sampled_neg_doc:
                bert_dict = tokenizer(query, doc_df.loc[nr_doc, 'doc_text'],
                                    max_length=512,
                                    padding='max_length',
                                    truncation=True) # dict of tensor {ids:[]...}
                bert_dict['q_id'] = q_id
                bert_dict['doc_id'] = nr_doc
                bert_dict['label'] = 0
                bert_data += [bert_dict]
            
            # break
        # print(bert_data)
        # break
    return bert_data # List[Dict[List]] = List[tokenizer output]


# ### training

# In[ ]:


train_bert_data = df_2_bert("train", train_q_df)


# In[ ]:


torch.save(train_bert_data, "./dataset/bert_data.pt")


# In[8]:


train_bert_data = torch.load("./dataset/1+3_bert_data.pt")


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

        # print(tokens_tensors, tokens_tensors.shape)
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
        
    # 計算分類準確率
    # _, binary_acc, bio_acc = get_predictions(model, trainLoader, compute_acc=True)
    timestamp(f"[epoch {epoch+1}] loss: {running_loss:.3f}")


# ## Model Inference

# In[ ]:





# ## Output

# In[ ]:




