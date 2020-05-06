import xml.etree.ElementTree as ElementTree
from pathlib import Path
import sys, os, IPython
from itertools import zip_longest

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

    self.train_data_path = Path("../dataset/pan19-author-profiling-training-2019-02-18/en")
    self.train_labels_path = Path("../dataset/pan19-author-profiling-training-2019-02-18/en_labels/truth.txt")
    self.test_data_path = Path("../dataset/pan19-author-profiling-test-2019-04-29/en")
    self.test_labels_path = Path("../dataset/pan19-author-profiling-test-2019-04-29/truth.txt")


  def get_data(self):

    train_data = {}; test_data = {}; train_labels = {}; test_labels = {}
    train_files = os.listdir(self.train_data_path)
    test_files = os.listdir(self.test_data_path)

    for train_file in train_files:
      path = os.path.join(self.train_data_path, train_file)
      root = ElementTree.parse(path).getroot()
      train_file = train_file[:-4] ## get rid of .xml suffix
      train_data[train_file] = [document.text for document in root.findall('documents/')]

    for test_file in test_files:
      path = os.path.join(self.test_data_path, test_file)
      root = ElementTree.parse(path).getroot()
      test_file = test_file[:-4] ## get rid of .xml suffix
      test_data[test_file] = [document.text for document in root.findall('documents/')]
    
    with open(self.train_labels_path, 'r') as fp_train, open(self.test_labels_path) as fp_test: 
      train_labels_data = fp_train.read().split('\n')[:-1]
      test_labels_data = fp_test.read().split('\n')[:-1]

      for row_train, row_test in zip_longest(train_labels_data, test_labels_data):
        train_file, train_label, train_gender = row_train.split(':::')
        train_labels[train_file] = (train_label, train_gender)
        if row_test is None: continue
        test_file, test_label, test_gender = row_test.split(':::')
        test_labels[test_file] = (test_label, test_gender)

    return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":

  ds = Dataset('a', 'a', 'a', 'a')
  a,b,c,d = ds.get_data()

  # for k in a:
  #   single_tweet = a[k][0]; break

  # print(f"TWEET: {single_tweet}")
  
  # nlp = en_core_web_sm.load()
  # doc = nlp(single_tweet)
  # spacy_words = [token.text for token in doc]
  # print(f"SPACY Tokenized words: {spacy_words}")

  # nltk_words = word_tokenize(single_tweet)
  # print(f"NLTK Tokenized words: {nltk_words}")