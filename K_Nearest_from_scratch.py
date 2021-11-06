import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
style.use('fivethirtyeight') #'ggplot'
import random
import pandas as pd
from collections import Counter

def k_nearest_neighbors(data, predict, k=3):
    if len(data) > k:
        warnings.warn("Value of k is lesser than total voting groups!!")
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result =  Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence

df = pd.read_csv(r'F:\Machine_Learning\breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

total = 0
correct = 0

for group_test in test_set:
    for predict_data in test_set[group_test]:
        result, confidence = k_nearest_neighbors(train_set, predict_data, k=5)
        if group_test == result:
            correct += 1
        total += 1
print('accuracy: ', correct/total)