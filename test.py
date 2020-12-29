import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-scratch", type=int, required=True)
parser.add_argument("-A", type=float)
parser.add_argument("-bm", type=float, default=1)
parser.add_argument("-test_from", type=str, required=True, help="model or bert score")
parser.add_argument("-mode", type=str, choices=["dev", "test", "alpha"], required=True)
args = parser.parse_args()

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(str(dt2)[:-13] + '\t' + msg)

LM = "bert-base-uncased"
if args.mode=="dev" or args.mode=="alpha":
    train_q_df = pd.read_csv("./dataset/train_queries.csv")
    dev_df = train_q_df[:20]
else:
    test_q_df = pd.read_csv("./dataset/test_queries.csv")

if args.scratch:
    if args.mode=="dev":
        bert_data = torch.load("./dataset/dev_bert_data.pt")
        bert_data = bert_data[:20000] # first 20st query 
    else:
        bert_data = torch.load("./dataset/test_bert_data.pt")

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    class QD_PairDataset(Dataset):
        def __init__(self, mode, list_of_bert):
            assert mode in ["train", "test", "dev"]
            self.mode = mode
            self.list_of_bert = list_of_bert
        def __getitem__(self, idx):
            bert_dict = self.list_of_bert[idx] #(batch=4; 1 pos 3 neg)
            inputid = torch.tensor([bert_dict['input_ids']]) #####
            tokentype = torch.tensor([bert_dict['token_type_ids']])#####
            attentionmask = torch.tensor([bert_dict['attention_mask']])#####
            q_id = bert_dict['q_id']
            doc_id = bert_dict['doc_id']

            return inputid, tokentype, attentionmask, q_id, doc_id

        def __len__(self):
            return len(self.list_of_bert)

    from transformers import BertForMultipleChoice
    model = BertForMultipleChoice.from_pretrained(LM, return_dict=True)
    # from transformers import BertForSequenceClassification
    # model = BertForSequenceClassification.from_pretrained(LM, return_dict=True)

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
    MODEL_PATH = args.test_from
    model.load_state_dict(torch.load(MODEL_PATH))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    BATCH_SIZE = 100
    testSet = QD_PairDataset("test", bert_data)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE)

    predictions = get_predictions(model, testLoader, BATCH_SIZE)

    if args.mode=="dev" or args.mode=="alpha":
        E = args.test_from[args.test_from.find("E_")+2 : -3]
        torch.save(predictions, f"./model/score/dev_bert_score_E{int(E)}.pt")####
    else:
        torch.save(predictions, "./model/score/test_bert_score.pt")
else:
    predictions = torch.load(args.test_from)

from ml_metrics import mapk
import numpy as np
A = args.A
bm = args.bm

if args.mode=="dev":
    q_list = dev_df['query_id']
    doc_list = dev_df['bm25_top1000']
    doc_score = dev_df['bm25_top1000_scores']
    pos_doc_ids = dev_df['pos_doc_ids']

    ans = [ids.split() for ids in pos_doc_ids]

    pre = []
    for i, q_id in enumerate(q_list):
        p = []
        d_list = doc_list[i].split()
        bm_score = np.array([float(s) for s in doc_score[i].split()])
        bert_score = []
        for j in range(1000):
            if q_id != predictions[i*1000+j]['q_id']:
                print(i, j, q_id, predictions[i*1000+j]['q_id'])
                exit(-1)
            if d_list[j] != predictions[i*1000+j]['doc_id']:
                print(i, j, d_list[j], predictions[i*1000+j]['doc_id'])
                exit(-1)
            bert_score += [predictions[i*1000+j]['score']]
        bert_score = np.array(bert_score)

        score = bm*bm_score + A*bert_score
        sortidx = np.argsort(score)
        sortidx = np.flip(sortidx)
        for idx in sortidx:
            p += [d_list[idx]]
        pre += [p]
    print(mapk(ans, pre, 1000))

elif args.mode == "test":
    q_list = test_q_df['query_id']
    doc_list = test_q_df['bm25_top1000']
    doc_score = test_q_df['bm25_top1000_scores']

    with open("result.csv", 'w') as fp:
        fp.write("query_id,ranked_doc_ids\n")
        for i, q_id in tqdm(enumerate(q_list)):
            d_list = doc_list[i].split()
            fp.write(str(q_id)+',')
            bm_score = np.array([float(s) for s in doc_score[i].split()])
            bert_score = []
            for j in range(1000):
                if q_id != predictions[i*1000+j]['q_id']:
                    print(i, j, q_id, predictions[i*1000+j]['q_id'])
                    exit(-1)
                if d_list[j] != predictions[i*1000+j]['doc_id']:
                    print(i, j, d_list[j], predictions[i*1000+j]['doc_id'])
                    exit(-1)
                bert_score += [predictions[i*1000+j]['score']]
            bert_score = np.array(bert_score)
            score = bm*bm_score + A*bert_score
            sortidx = np.argsort(score)
            sortidx = np.flip(sortidx)
            
            for idx in sortidx:
                fp.write(d_list[idx]+' ')
            fp.write("\n")
    timestamp("output done")

elif args.mode=="alpha":
    q_list = dev_df['query_id']
    doc_list = dev_df['bm25_top1000']
    doc_score = dev_df['bm25_top1000_scores']
    pos_doc_ids = dev_df['pos_doc_ids']
    ans = [ids.split() for ids in pos_doc_ids]
    map_score = []
    bm = 1
    step = 0.01

    for A in tqdm(np.arange(0.0, args.A, step)):
        pre = []
        for i, q_id in enumerate(q_list):
            p = []
            d_list = doc_list[i].split()
            bm_score = np.array([float(s) for s in doc_score[i].split()])
            bert_score = []
            for j in range(1000):
                if q_id != predictions[i*1000+j]['q_id']:
                    print(i, j, q_id, predictions[i*1000+j]['q_id'])
                    exit(-1)
                if d_list[j] != predictions[i*1000+j]['doc_id']:
                    print(i, j, d_list[j], predictions[i*1000+j]['doc_id'])
                    exit(-1)
                bert_score += [predictions[i*1000+j]['score']]
            bert_score = np.array(bert_score)

            sortidx = np.argsort(score)
            sortidx = np.flip(sortidx)
            for idx in sortidx:
                p += [d_list[idx]]
            pre += [p]
        map_score += [mapk(ans, pre, 1000)]
    map_score = np.array(map_score)
    plt.plot(map_score)
    plt.savefig('myfig1.png')
    print(f"alpha={np.argmax(map_score)*step}\t{map_score.max()}")