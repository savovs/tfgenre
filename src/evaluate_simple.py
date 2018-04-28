import tflearn
from models import simple
from data import test_x, test_y, categories

simple_model = tflearn.DNN(simple(len(categories)))
simple_model.load('../trained_models/simple/simple_model')
accuracy = simple_model.evaluate(test_x, test_y)

print('Simple CNN accuracy: ', accuracy)
