<p style="text-align:right;">
姓名:李韋宗<br>
學號:B10615024<br>
日期:2021/1/15<br>
</p>

<h1 style="text-align:center;"> Homework 6: Transformer-based Models

## 資料前處理
* 2 種模型預處理方法一致
* 如同 baseline 作法，用 tokienizer 截斷至長度 512
* `[CLS]` query `[SEP]` document `[SEP]`

## 模型參數調整
* Pretrained lenguage model: bert-base-uncased
* Optimizer: AdamW 
* Learning rate = 1e-5
* Split 20 documents of training queries to grid search optimal $\alpha$ for BERT

### multiple choice
* Batch size = 3
* Num. epochs = 2
* Num. of negative documents = 3
* $\alpha$ from greedy search = 1.13

### regression
* Batch size = 2
* Num. epochs = 1
* Num. of negative documents = 3
    * Num. of high BM score documents = 1
    * Num. of low BM score documents = 2
* $\alpha$ from greedy search = 0.59

## 模型運作原理
### multiple choice
* 如同 baseline 用 BERT 做多選題任務

### regression
* 新設定一個分數用 BERT 做評分任務
* 1 篇 positve 配上 1 篇分數較高的 negetive 和 2 篇分數極低的 negative
* positive 的 label 設為 50 (因為 50 可以超過所有 BM 算出來的分數)
* negative 的 label 同為 BM25 分數
* 通過 BERT 後再通過 2 層 linear (效果比 1 層好)
    * 維度: BERT hidden_size(768) -> 200 -> 1
* 最終分數一樣為: $𝑠𝑐𝑜𝑟e_{BM25} + \alpha ⋅ 𝑠𝑐𝑜𝑟e_{BERT}$

## 心得
* 終於使用到 BERT 了，但最開始還是無法突破 baseline，還試了同學間的不同的 pointwise 方法但依舊無果，看了助教在討論區的回覆才發現輸出的地方出了愚蠢的嚴重錯誤，所以分數超低，修正之後就馬上過了。接著思考如何改進，如同上述嘗試使用 regression 解這個問題，但分數沒有突破 baseline，但因期末實在過於忙碌就作罷，最終仍使用 multiple choice 模型。撰寫報告的現在，發現 regression 可能因為 2 層 linear 間只加了 dropout，忘了加 activation function，所以表現不佳。

PS 這邊說的 baseline 是比較高的那條，老早就做完了，但一直拖延報告到最後一刻 OAO

## 參考資料
> 助教 baseline 的方法

> huggingface document
>> https://huggingface.co/transformers/

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
