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
parser.add_argument("-model_name", type=str, required=True) # mul
parser.add_argument("-bert_data_path", type=str, required=True) # "./dataset/1+3_bert_data.pt"
parser.add_argument("-mode", type=str, choices=["train", "dev"], required=True)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-lr", type=float, default=3e-5)
parser.add_argument("-epoch", type=int, default=2)
parser.add_argument('-train_from', type=str, default="")
args = parser.parse_args()

if args.mode == "dev":
    args.model_name = "dev_" + args.model_name

train_bert_data = torch.load(args.bert_data_path)
if args.mode == "dev":
    train_bert_data = train_bert_data[1521:] # after 21st query 

import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class QD_PairDataset(Dataset):
    def __init__(self, mode, list_of_bert):
        self.mode = mode
        self.list_of_bert = list_of_bert

    def __getitem__(self, idx):
        bert_dict = self.list_of_bert[idx] #(batch=4; 1 pos 3 neg)
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
    def __len__(self):
        return len(self.list_of_bert)

from transformers import BertForMultipleChoice
model = BertForMultipleChoice.from_pretrained(args.LM, return_dict=True)

""" model setting (training)"""
from transformers import BertConfig, AdamW
trainSet = QD_PairDataset("train", train_bert_data)##########
trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
optimizer = AdamW(model.parameters(), lr=args.lr) # AdamW = BertAdam
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
train_from = 0
EPOCHS = args.epoch
if args.train_from != "":
    model.load_state_dict(torch.load(args.train_from))
    train_from = int(args.train_from[args.train_from.find("E_")+2 : -3])

model = model.to(device)
model.train()
timestamp("start training")
for epoch in range(train_from, EPOCHS):
    running_loss = 0.0
    acc = 0.0
    c = 0
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
        acc += torch.count_nonzero(outputs.logits.argmax(axis=1)==labels).item() / labels.shape[0]
        c += 1
        # backward
        loss.backward()
        optimizer.step()

        # 紀錄當前 batch loss
        running_loss += loss.item()
    #     break
    # break
    
    CHECKPOINT_NAME = f"./model/{args.model_name}_{args.LM.replace('-', '_')}_E_{str(epoch+1)}.pt"
    torch.save(model.state_dict(), CHECKPOINT_NAME)
    timestamp(f"[epoch {epoch+1}] loss: {running_loss:.3f} acc: {acc/c}")