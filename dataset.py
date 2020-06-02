import xml.etree.ElementTree as ElementTree
from pathlib import Path
import sys, os, IPython
from itertools import zip_longest

import torch.utils.data 
import torch
from torch.nn.utils.rnn import pad_sequence

# from nltk.tokenize import sent_tokenize, word_tokenize
# import spacy
# import en_core_web_sm

# import nltk
# nltk.download('punkt')


class Dataset:

    """
    params: 
    train_data_path : str (relative path to training data dir)
    train_labels_path : str (relative path to training labels 'truth.txt')
    test_data_path : str (relative path to test data dir)
    test_labels_path: str (relative path to test labels 'truth.txt')
  """

    def __init__(self, train_data_path, train_labels_path, test_data_path, test_labels_path):

        self.train_data_path = train_data_path
        self.train_labels_path = train_labels_path
        self.test_data_path = test_data_path
        self.test_labels_path = test_labels_path

    def get_data(self):

        train_data = {}
        test_data = {}
        train_labels = {}
        test_labels = {}
        train_files = os.listdir(self.train_data_path)
        test_files = os.listdir(self.test_data_path)

        for train_file in train_files:
            path = os.path.join(self.train_data_path, train_file)
            root = ElementTree.parse(path).getroot()
            train_file = train_file[:-4]  ## get rid of .xml suffix
            train_data[train_file] = [
                document.text for document in root.findall("documents/")
            ]

        for test_file in test_files:
            path = os.path.join(self.test_data_path, test_file)
            root = ElementTree.parse(path).getroot()
            test_file = test_file[:-4]  ## get rid of .xml suffix
            test_data[test_file] = [
                document.text for document in root.findall("documents/")
            ]

        with open(self.train_labels_path, "r") as fp_train, open(
            self.test_labels_path
        ) as fp_test:
            train_labels_data = fp_train.read().split("\n")[:-1]
            test_labels_data = fp_test.read().split("\n")[:-1]

            for row_train, row_test in zip_longest(train_labels_data, test_labels_data):
                train_file, train_label, train_gender = row_train.split(":::")
                train_labels[train_file] = (train_label, train_gender)
                if row_test is None:
                    continue
                test_file, test_label, test_gender = row_test.split(":::")
                test_labels[test_file] = (test_label, test_gender)

        return train_data, train_labels, test_data, test_labels


class DatasetPyTorch(torch.utils.data.Dataset):

    embedding_size = 0

    def __init__(self, dataset, embeddings, embedding_source='glove'):
        super(DatasetPyTorch, self).__init__()
        self.dataset = dataset
        self.embeddings = embeddings
        self.embedding_source = embedding_source
        DatasetPyTorch.embedding_size = 49 if embedding_source == 'glove' else 768

    def __getitem__(self, index):
        if self.embedding_source == 'glove':
            tweet_tokens = self.dataset['clean_tweet'][index]
            tweet_embedding = torch.zeros(len(tweet_tokens), DatasetPyTorch.embedding_size)
            for i in range(tweet_embedding.size()[0]):
                tweet_embedding[i] = self.embeddings.get(tweet_tokens[i], self.embeddings['<UNK>'])
            #tweet_embedding = torch.as_tensor([self.embeddings.get(token, self.embeddings['<UNK>']) for token in tweet_tokens])
            label = torch.Tensor([1.]) if self.dataset['bot'][index] == "bot" else torch.Tensor([0.])
            return (tweet_embedding, label)
        else:
            account = self.dataset['author'][index]
            account_index_first = self.dataset[self.dataset.author == account].index[0]
            tweet_embedding_index = index - account_index_first

            objects = self.embeddings[account]
            objects = list(objects)
            tweet = objects[tweet_embedding_index] # 1, 768

            tweet_embedding = torch.as_tensor([float(x) for x in tweet])
            label = torch.Tensor([1.]) if self.dataset['bot'][index] == "bot" else torch.Tensor([0.])
            tweet_embedding = torch.unsqueeze(tweet_embedding, 0)
            return (tweet_embedding, label)
    
    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """

    tweets, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in tweets]) # Needed for later
    #print(texts); print(labels); print(lengths)
    #print(type(tweets[0][0])); print(len(tweets[0][0]))
    padded_tweets = pad_sequence(tweets, batch_first=True, padding_value=0)
    labels = torch.as_tensor(labels)
    #labels = torch.Tensor(labels)
    #print(padded_texts)
    # Process the text instances
    return padded_tweets, labels

if __name__ == "__main__":

    ds = Dataset(
        Path(
            "/users/ianic/tar/pan19/pan19-author-profiling-training-2019-02-18/en"
        ),
        Path(
            "/users/ianic/tar/pan19/pan19-author-profiling-training-2019-02-18/en_labels/truth.txt"
        ),
        Path("/users/ianic/tar/pan19/pan19-author-profiling-test-2019-04-29/en"),
        Path(
            "/users/ianic/tar/pan19/pan19-author-profiling-test-2019-04-29/truth.txt"
        ),
    )
    a, b, c, d = ds.get_data()

    # for k in a:
    #   single_tweet = a[k][0]; break

    # print(f"TWEET: {single_tweet}")

    # nlp = en_core_web_sm.load()
    # doc = nlp(single_tweet)
    # spacy_words = [token.text for token in doc]
    # print(f"SPACY Tokenized words: {spacy_words}")

    # nltk_words = word_tokenize(single_tweet)
    # print(f"NLTK Tokenized words: {nltk_words}")

