import tflearn
from models import google_net
from data import test_x, test_y, categories

model = tflearn.DNN(google_net(len(categories)))
model.load('../trained_models/google_net/google_net_model')
accuracy = model.evaluate(test_x, test_y)

print('GoogleNet accuracy: ', accuracy)
