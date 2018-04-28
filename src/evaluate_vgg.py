import tflearn
from models import vgg
from data import test_x, test_y, categories

vgg_model = tflearn.DNN(vgg(len(categories)))
vgg_model.load('../trained_models/vgg/vgg_model')
accuracy = vgg_model.evaluate(test_x, test_y)

print('VGG accuracy: ', accuracy)
