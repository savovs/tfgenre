# import tensorflow as tf
# import tflearn
from ast import literal_eval
import pandas as pd
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/data.csv'
data = pd.read_csv(data_path)

for index, item in data.iterrows():
    mfcc = literal_eval(item['lowLevel.mfcc'])
    print(index, len(mfcc), len(mfcc[0]))