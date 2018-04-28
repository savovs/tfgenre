import tflearn
from models import alex_net
from data import test_x, test_y, categories

alex_net_model = tflearn.DNN(alex_net(len(categories)))
alex_net_model.load('../trained_models/alex_net/alex_net_model')
accuracy = alex_net_model.evaluate(test_x, test_y)

print('AlexNet accuracy: ', accuracy)
