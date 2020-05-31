from pathlib import Path
import sys, random
from collections import defaultdict
import numpy as np
import pandas as pd
import json
import IPython

import spacy
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import Dataset, DatasetPyTorch, collate_fn
from glove import Glove
from models import RNNModel
from evaulate import evaluate_rnn, eval_perf_binary


glove_path = Path('/home/ivanbilic/fer/tar/dataset/glove.twitter.27B/glove.twitter.27B.50d.txt')
# bert_path = Path('/home/ivanbilic/fer/tar/dataset/bert_embeddings_train.json')
# import ijson
# for prefix, the_type, value in ijson.parse(open(bert_path)):
#     print (prefix, the_type, value)

# with open(bert_path, 'r') as f:
    # objects = ijson.items(f, "7fbb9ceb600ebc6fcadc9ee235cda580.item")
    # cnt = 0
    #tweet = [tweet for idx, tweet in enumerate(objects) if idx == 90]
    #print(tweet)
    # for tweet in objects:
        # if cnt != 90: print(cnt); cnt += 1; continue
        # tweet = [float(x) for x in tweet]
        # break
# print(tweet); print(len(tweet))
        


df_train = pd.read_pickle("pan19_df_clean_train_glove.pkl")
df_test = pd.read_pickle("pan19_df_clean_test_glove.pkl")
print(type(df_train['clean_tweet'][0])); print(df_train['clean_tweet'][0]); print(len(df_train)); print(df_train['bot'][0])

######## extra preproc: stopwords, <rt>, lemmas #########
# tokenizer = TweetTokenizer()
#lemmatizer = WordNetLemmatizer() 
# for i in range(len(df_train['clean_tweet'])):
    # type(df_train['clean_tweet'][i])
    #print(i)
    #tokenized = tokenizer.tokenize(df_train['clean_tweet'][i])
    # tokenized = ['<rt>' if token == 'rt' else token for token in tokenized] ##add <rt> token since it exists in glove
    #tokenized = [token for token in tokenized if not token in stopwords.words('english')] ##remove stopwords
    #print(tokenized)
    #tokenized = list(map(lambda x : lemmatizer.lemmatize(x), tokenized))

lr=1e-4; num_epoch=5; batch_size_train = 64; batch_size_test = 32; gradient_clip=0.25
torch.manual_seed(7052020)
np.random.seed(7052020)

########################## choose embeddings ##########################
embedding_source = 'glove'
embeddings = Glove(glove_path).glove_dict
# embedding_source = 'bert'
# embeddings = Path('/home/ivanbilic/fer/tar/dataset/bert_embeddings_train.json')
#print(len(embeddings)); print(embeddings['<user>']); print(type(embeddings['<user>'])); print(embeddings['<user>'].size()); 

train_dataset = DatasetPyTorch(dataset=df_train, embeddings=embeddings, embedding_source=embedding_source)
bert_valid_dataset = DatasetPyTorch(dataset=df_train[:500], embeddings=embeddings, embedding_source=embedding_source)
test_dataset = DatasetPyTorch(dataset=df_test, embeddings=embeddings, embedding_source=embedding_source)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
bert_valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, num_workers=2, collate_fn=collate_fn, drop_last=False)

model = RNNModel(cell_type='lstm', embedding_size=DatasetPyTorch.embedding_size, num_layers=2, bidirectional=False, dropout_prob=0)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    model.train()
    train_iter = iter(train_loader)
    num_train_batches = len(train_iter)
    print(num_train_batches)
    batch_index = 0
    while batch_index < num_train_batches:
        data, labels, lengths = train_iter.next()
        if batch_index % 1000 == 0: print(batch_index)
        #print(len(data), len(data[0]), len(data[-1]))
        #print(labels)
        #IPython.embed()
        #sys.exit()
        data = torch.transpose(data, 0, 1) #time-first format, speed gain
        logits = model(data)
        loss = criterion(logits, labels)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_index += 1
    if epoch % 1 == 0:
        preds, ground_truth, loss_avg = evaluate_rnn(model, criterion, optimizer, bert_valid_loader)
        accuracy, recall, precision, f1 = eval_perf_binary(preds, ground_truth)
        print(f"epoch: {epoch}, loss_avg: {np.mean(loss_avg)}, acc: {accuracy}, recall: {recall}, precision: {precision}, f1: {f1}")


preds, ground_truth, loss_avg = evaluate_rnn(model, criterion, optimizer, test_loader)
accuracy, recall, precision, f1 = eval_perf_binary(preds, ground_truth)
print(f"test accuracy {accuracy}, f1: {f1}, recall: {recall}, precision: {precision}")






