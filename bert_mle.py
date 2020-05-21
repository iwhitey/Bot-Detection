from pathlib import Path
from sentence_transformers import SentenceTransformer
import spacy
from dataset import Dataset
import sys, random
from spacy.lang.en import English
from collections import defaultdict
import numpy as np

model = SentenceTransformer('bert-base-nli-mean-tokens')
dataset = Dataset(
        Path("/home/ivanbilic/fer/tar/dataset/pan19-author-profiling-training-2019-02-18/en"),
        Path("/home/ivanbilic/fer/tar/dataset/pan19-author-profiling-training-2019-02-18/en_labels/truth.txt"),
        Path("/home/ivanbilic/fer/tar/dataset/pan19-author-profiling-test-2019-04-29/en"),
        Path("/home/ivanbilic/fer/tar/dataset/pan19-author-profiling-test-2019-04-29/truth.txt"),
    )
train_data, train_labels, test_data, test_labels = dataset.get_data()

#rule based senencizer works better on our twitter data than the dependency parser based sentence segmentizer
nlp = English()  # just the language with no model
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)
bot_count = 0; human_count = 0; tweet_embeddings = defaultdict(list) 
for cnt, account in enumerate(train_data):
    print(f"account: {cnt}")

    ################## uncomment code to take only N humans and N bots (N=30 here) ###################
    # account = random.choice(list(train_data.keys()))
    # if bot_count == 30 and human_count == 30: break
    # if bot_count == 30 and train_labels[account][0] == "bot": continue
    # if human_count == 30 and train_labels[account][0] == "human": continue
    # if train_labels[account][0] == "bot" and bot_count < 30: print("bot"); bot_count += 1
    # if train_labels[account][0] == "human" and human_count < 30: print("human"); human_count += 1
    
    tweets = train_data[account]
    for i, tweet in enumerate(tweets):
        doc = nlp(tweet)
        sentences = [str(sentence) for sentence in doc.sents]
        sentence_embeddings = model.encode(sentences)
        tweet_embed = 0
        for j in range(len(sentence_embeddings)):
            tweet_embed += sentence_embeddings[j] / len(sentence_embeddings)
        tweet_embeddings[account].append(tweet_embed)


####### estimate mean and variances per account, calculate mean variances (per feature) for bots and humans, and compare bot and human mean variance
keys_list = list(tweet_embeddings.keys())
bots_keys = [key for key in keys_list if train_labels[key][0] == "bot"]
humans_keys = [key for key in keys_list if train_labels[key][0] == "human"]
print(len(bots_keys), len(humans_keys))

bots_diags = []
for bot in bots_keys:
    bot_data = np.array(tweet_embeddings[bot]).reshape(100, 768)
    mean_bot = np.mean(bot_data, axis=0)
    cov_bot = (bot_data - mean_bot).T @ (bot_data - mean_bot) / (bot_data.shape[0] - 1)
    bot_diag = cov_bot.diagonal()
    bots_diags.append(bot_diag)

humans_diags = []
for human in humans_keys:
    human_data = np.array(tweet_embeddings[human]).reshape(100, 768)
    mean_human = np.mean(human_data, axis=0)
    cov_human = (human_data - mean_human).T @ (human_data - mean_human) / (human_data.shape[0] - 1)
    human_diag = cov_human.diagonal()
    humans_diags.append(human_diag)

humans_diags = np.array(humans_diags); bots_diags = np.array(bots_diags)
mean_bots_diag = np.mean(bots_diags, axis=0)
mean_humans_diag = np.mean(humans_diags, axis=0)

print(f"BOTS MEAN DIAG VARIANCE: {mean_bots_diag}")
print(f"HUMANS MEAN DIAG VARIANCE: {mean_humans_diag}")
print(f"HUMANS MEAN DIAG VARIANCE GREATER THAN BOTS MEAN DIAG VARIANCE FOR: {mean_humans_diag[mean_humans_diag > mean_bots_diag].shape[0]} dims")
print(f"BOTS MEAN of MEAN DIAG VARIANCE {np.mean(mean_bots_diag)}")
print(f"HUMANS MEAN of MEAN DIAG VARIANCE {np.mean(mean_humans_diag)}")

############ estimate mean and variance as if all bot tweets and human tweets come from the same distribution (account)##############
# bots_data = np.array([tweet_embeddings[k] for k in bots_keys])
# humans_data = np.array([tweet_embeddings[k] for k in humans_keys])
# print(len(humans_data), len(bots_data))
# bots_data = bots_data.reshape(-1, 768)
# humans_data = humans_data.reshape(-1, 768)

# mean_bots = np.mean(bots_data, axis=0)
# cov_bots = (bots_data - mean_bots).T @ (bots_data - mean_bots) / (bots_data.shape[0] - 1)
# mean_humans = np.mean(humans_data, axis=0)
# cov_humans = (humans_data - mean_humans).T @ (humans_data - mean_humans) / (humans_data.shape[0] - 1)

# #print(f"variance_humans: {cov_humans.diagonal()}")
# #print(f"variance_bots: {cov_bots.diagonal()}")
# bots_diag = cov_bots.diagonal()
# humans_diag = cov_humans.diagonal()

# comparison = bots_diag[bots_diag > humans_diag].shape
# print(comparison)
