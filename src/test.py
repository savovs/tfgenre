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
data = pd.read_csv(data_path)

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


print(train_y)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# print(type(train_x), type(train_y), type(test_x), type(test_y))

net = input_data(shape=[None, 1300, 13, 1], name='input')
net = conv_2d(net, 32, 2, activation='relu')
net = max_pool_2d(net, 2)

net = conv_2d(net, 64, 2, activation='relu')
net = max_pool_2d(net, 2)

net = fully_connected(net, 1024, activation='relu')
net = dropout(net, 0.8)
net = tflearn.fully_connected(net, len(categories), activation='softmax')
net = regression(net, learning_rate=0.01, loss='categorical_crossentropy', name='targets')

# Define model
model = tflearn.DNN(net)

# Train using gradient descent
model.fit(
    { 'input': train_x },
    { 'targets': train_y },
    n_epoch=10,
    validation_set=({ 'input': test_x }, { 'targets': test_y }),
    snapshot_step=500,
    show_metric=True,
    run_id='genres'
)
