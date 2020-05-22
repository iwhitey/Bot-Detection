from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy
from dataset import Dataset
import sys, random
from spacy.lang.en import English
from collections import defaultdict
import numpy as np
import json

model = SentenceTransformer('bert-base-nli-mean-tokens')
dataset = Dataset(Path("/home/ianic/tar/pan19-author-profiling-training-2019-02-18/en"),
  Path("/home/ianic/tar/pan19-author-profiling-training-2019-02-18/en_labels/truth.txt"),
  Path("/home/ianic/tar/pan19-author-profiling-test-2019-04-29/en"),
  Path("/home/ianic/tar/pan19-author-profiling-test-2019-04-29/truth.txt"))

train_data, train_labels, test_data, test_labels = dataset.get_data()

print(len(train_data), len(train_labels))
print(len(test_data), len(test_labels))

nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
tweet_embeddings = defaultdict(list) 

for cnt, account in enumerate(train_data):
    print(f"account: {cnt}")
    tweets = train_data[account]
    for i, tweet in enumerate(tweets):
        doc = nlp(tweet)
        sentences = [str(sentence) for sentence in doc.sents]
        sentence_embeddings = model.encode(sentences)
        tweet_embed = 0
        for j in range(len(sentence_embeddings)):
            tweet_embed += sentence_embeddings[j] / len(sentence_embeddings)
        tweet_embeddings[account].append(tweet_embed.tolist())

with open('bert_embeddings_train.json', 'w') as fp:
    json.dump(tweet_embeddings, fp)

# with open('bert_embeddings.json', 'r') as fp:
#     tweet_embeddings = json.load(fp)
#     print(type(tweet_embeddings))
#     print(type(tweet_embeddings[account]))
#     print(len(tweet_embeddings[account]))
#     print(len(tweet_embeddings[account][0]))