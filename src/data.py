from tflearn.data_utils import to_categorical

from ast import literal_eval
import pandas as pd
import numpy as np
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/data.csv'
data = pd.read_csv(data_path)

# Parse strings and make arrays the same shape
data['lowLevel.mfcc'] = data['lowLevel.mfcc'].map(literal_eval)
data['lowLevel.mfcc'] = data['lowLevel.mfcc'].map(np.array)
data['lowLevel.mfcc'] = data['lowLevel.mfcc'].map(lambda item: np.resize(item, (1300, 13)))

# Categories to numbers
categories = list(data['category'].unique())
num_categories = len(categories)

data['category'] = data['category'].map(categories.index)

train = data.sample(frac=0.8)
test = data.drop(train.index)

train_x = np.empty((train['lowLevel.mfcc'].size, 1300, 13))
test_x = np.empty((test['lowLevel.mfcc'].size, 1300, 13))

for index, item in enumerate(train['lowLevel.mfcc']):
    train_x[index] = item

for index, item in enumerate(test['lowLevel.mfcc']):
    test_x[index] = item

train_y = to_categorical(train['category'], len(categories))
test_y = to_categorical(test['category'], len(categories))

train_x = train_x.reshape([-1, 1300, 13, 1])
test_x = test_x.reshape([-1, 1300, 13, 1])