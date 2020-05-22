import nltk
import os
import pandas as pd
import re
import sys
from dataset import Dataset
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from pathlib import Path

"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu (github.com/tokestermw)

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

this version from gist.github.com/ppope > preprocess_twitter.py

light edits by amackcrane, mostly inspired by the test case given at bottom
"""

import sys
import regex as re

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> " # amackcrane added trailing space


def tokenize(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\w+", hashtag)  # amackcrane edit
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    #text = re_sub(r"([A-Z]){2,}", allcaps)  # moved below -amackcrane

    # amackcrane additions
    text = re_sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2")
    text = re_sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )")
    text = re_sub(r"  ", r" ")
    text = re_sub(r" ([A-Z]){2,} ", allcaps)
    
    return text.lower()

if __name__ == '__main__':
    
    dataset = Dataset(Path("/home/ianic/tar/pan19-author-profiling-training-2019-02-18/en"),
      Path("/home/ianic/tar/pan19-author-profiling-training-2019-02-18/en_labels/truth.txt"),
      Path("/home/ianic/tar/pan19-author-profiling-test-2019-04-29/en"),
      Path("/home/ianic/tar/pan19-author-profiling-test-2019-04-29/truth.txt"))

    xtrain, ytrain, xtest, ytest = dataset.get_data()
    d = {'author': [], 'tweet': [], 'bot': []}
    for author in xtrain:
        for tweet in xtrain[author]:
            d['author'].append(author)
            d['tweet'].append(tweet)
            d['bot'].append(ytrain[author][0])

    df_train = pd.DataFrame(d, columns=['author', 'tweet', 'bot'])

    d = {'author': [], 'tweet': [], 'bot': []}
    for author in xtest:
        for tweet in xtest[author]:
            d['author'].append(author)
            d['tweet'].append(tweet)
            d['bot'].append(ytest[author][0])

    df_test = pd.DataFrame(d, columns=['author', 'tweet', 'bot'])

    df_train['clean_tweet'] = df_train['tweet'].apply(lambda x: tokenize(x))
    df_test['clean_tweet'] = df_test['tweet'].apply(lambda x: tokenize(x))

    df_train.to_pickle("./pan19_df_clean_train_glove.pkl")
    df_test.to_pickle("./pan19_df_clean_test_glove.pkl")
        
    # #_, text = sys.argv  # kaggle envt breaks this -amackcrane
    # #if text == "test":
    # text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    # text2 = "TEStiNg some *tough* #CASES" # couple extra tests -amackcrane
    # tokens = tokenize(text)
    # print(tokens)
    # print(tokenize(text2))
