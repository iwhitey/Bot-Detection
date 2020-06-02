import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from time import time

import IPython
import nltk
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import Dataset, DatasetPyTorch, collate_fn
from evaulate import eval_perf_binary, evaluate_rnn
from glove import Glove
from models import RNNModel
#nltk.download('stopwords')

print(torch.cuda.is_available())   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"CUDA enabled device: {device}")

_time_start = time()
glove_path = Path('/home/ianic/tar/Bot-Detection/glove/glove.twitter.27B.50d.txt')
# bert_path_train = Path('/home/ianic/tar/Bot-Detection/bert_embeddings_train.json')
# bert_path_test = Path('/home/ianic/tar/Bot-Detection/bert_embeddings_test.json')
  
_df_train = pd.read_pickle("/home/ianic/tar/Bot-Detection/pan19_df_clean_train_glove.pkl")
_df_test = pd.read_pickle("/home/ianic/tar/Bot-Detection/pan19_df_clean_test_glove.pkl")

n_dev = 82400 # 20%

df_train = _df_train.head(-n_dev)
df_dev = _df_train.tail(n_dev) 
df_dev.reset_index(inplace=True)
df_test = _df_test

print(df_train.info())
print(df_dev.info())
print(df_test.info())

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

lr = 1e-4
num_epoch = 9
batch_size_train = 64
batch_size_test = 32
gradient_clip = 0.25
torch.manual_seed(7052020)
np.random.seed(7052020)

########################## choose embeddings ##########################
embedding_source = 'glove'
embeddings = Glove(glove_path).glove_dict
# embedding_source = 'bert'
# embeddings = Path(bert_path_train)
_time = time()

# bert_dict_train = None
# with open(Path(bert_path_train), 'r') as f:
#     bert_dict_train = json.load(f)
# print("Initialized train embeddings!")

# bert_dict_test = None
# with open(Path(bert_path_test), 'r') as f:
#     bert_dict_test = json.load(f)
# print("Initialized test embeddings!")

train_dataset = DatasetPyTorch(
    dataset=df_train, embeddings=embeddings, 
    embedding_source=embedding_source)
dev_dataset = DatasetPyTorch(
    dataset=df_dev, embeddings=embeddings, 
    embedding_source=embedding_source)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size_train, 
    shuffle=True, num_workers=32, 
    drop_last=False, 
    collate_fn=collate_fn)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=batch_size_train, 
    shuffle=True, num_workers=32, 
    drop_last=False, 
    collate_fn=collate_fn)

print(f"loading data: {time() - _time}")

model = RNNModel(
    cell_type='lstm', embedding_size=DatasetPyTorch.embedding_size, 
    num_layers=3, bidirectional=False, dropout_prob=0)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=lr)
print("training")
model = model.to(device)

# to track the training loss as the model trains
train_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average devation loss per epoch as the model trains
avg_dev_losses = [] 

for epoch in range(num_epoch):
    _time = time()
    model.train()
    train_iter = iter(train_loader)
    num_train_batches = len(train_iter)
    print(num_train_batches)
    batch_index = 0
    while batch_index < num_train_batches:
        data, labels = train_iter.next()
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        data = torch.transpose(data, 0, 1) #time-first format, speed gain bert-[1, 64, 768]
#         labels = labels.squeeze() uncomment for bert, comment for glove
            
        logits = model(data)
        loss = criterion(logits, labels)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_index += 1
        
        train_losses.append(loss.item())

    print(f"epoch time: {time() - _time}")
    _time = time()
    preds, ground_truth, loss_avg = evaluate_rnn(
        model, criterion, optimizer, dev_loader, device)
    
    train_loss = np.mean(train_losses)
    dev_loss = np.mean(loss_avg)
    avg_train_losses.append(train_loss)
    avg_dev_losses.append(dev_loss)
    train_losses = []
        
    accuracy, recall, precision, f1 = eval_perf_binary(preds, ground_truth)
    print(f"epoch: {epoch}, inference time:{time() - _time}, "
          f"loss_avg: {np.mean(loss_avg)}, acc: {accuracy}, "
          f"recall: {recall}, precision: {precision}, f1: {f1}")

test_dataset = DatasetPyTorch(
    dataset=df_test, embeddings=embeddings, 
    embedding_source=embedding_source)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size_test, 
    shuffle=True, num_workers=32,
    drop_last=False, 
    collate_fn=collate_fn)

preds, ground_truth, loss_avg = evaluate_rnn(
    model, criterion, optimizer, test_loader, device)
accuracy, recall, precision, f1 = eval_perf_binary(preds, ground_truth)

print(f"epochs: {num_epoch}, total execution time:{time() - _time_start}, "
      f"loss_avg: {np.mean(loss_avg)}, acc: {accuracy}, "
      f"recall: {recall}, precision: {precision}, f1: {f1}")

print("lstm, glove 50")
print(avg_train_losses)
print(avg_dev_losses)
