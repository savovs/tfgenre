import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences

from ast import literal_eval
import pandas as pd
import numpy as np
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/data.csv'
data = pd.read_csv(data_path, nrows=10)

# Parse strings and make arrays the same shape
data['lowLevel.mfcc'] = data['lowLevel.mfcc'].map(literal_eval)
data['lowLevel.mfcc'] = data['lowLevel.mfcc'].map(np.array)
data['lowLevel.mfcc'] = data['lowLevel.mfcc'].map(lambda item: np.resize(item, (1300, 13)))

# Categories to numbers
categories = list(data['category'].unique())
num_categories = len(categories)

data['category'] = data['category'].map(categories.index)

train = data.sample(frac=0.8, random_state=200)
test = data.drop(train.index)

train_x = train['lowLevel.mfcc'].as_matrix()
test_x = test['lowLevel.mfcc'].as_matrix()

train_y = to_categorical(train['category'], len(categories))
test_y = to_categorical(test['category'], len(categories))



print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
print(type(train_x), type(train_y), type(test_x), type(test_y))

train_x = train_x.reshape([-1, 1300, 13, 1])
test_x = test_x.reshape([-1, 1300, 13, 1])

# net = input_data(shape=[None, 1300, 13, 1], name='input')
# net = conv_2d(net, 32, 2, activation='relu')
# net = tflearn.fully_connected(net, 32)
# net = tflearn.fully_connected(net, 32)
# net = tflearn.fully_connected(net, 2, activation='softmax')
# net = tflearn.regression(net)

# # Define model
# model = tflearn.DNN(net)

# # Train using gradient descent
# model.fit(
#     train_x,
#     train_y,
#     n_epoch=10,
#     validation_set=(test_x, test_y),
#     show_metric=True,
#     run_id='genres'
# )
