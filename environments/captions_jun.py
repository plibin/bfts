import csv
from bandit import Bandit
import numpy as np
import os
import environments

#read the caption data and group the rewards per caption
def read_data(data, nr_captions, nr_categories):
    captions_data = np.zeros((nr_captions, nr_categories))
    with open(data,encoding="utf8", errors='ignore', newline='') as csvfile:
             reader = csv.DictReader(csvfile)
             for row in reader:
                 caption_id = int(row['target_id']) - 1
                 reward_id = int(row['target_reward']) - 1
                 count = captions_data[caption_id][reward_id] 
                 captions_data[caption_id][reward_id] = count + 1
    return captions_data

def captions_data():
    path = os.path.dirname(environments.__file__)
    captions_data = read_data(path + '/499-responses.csv', 499, 3)
    #normalize the data
    for row in captions_data:
        sum_row = np.sum(row)
        for i in range(len(row)):
            row[i] = row[i]/sum_row
    return captions_data

def captions_means(n):
    if n > len(captions_data()):
        raise ValueError("No more than ",len(captions_data()), " are available")
    categories=np.array([0,0.5,1])
    mean_fn = lambda i: np.dot(i, categories)
    means = list(map(mean_fn, captions_data()[:n]))
    return means

def captions_bandit(n):
    categories=np.array([0,0.5,1])
    data = captions_data()[:n]
    def reward_fn(percentage):
        return lambda: np.random.choice(categories,1,p=percentage)[0]
    arms = list(map(reward_fn, data))
    return Bandit(arms)
