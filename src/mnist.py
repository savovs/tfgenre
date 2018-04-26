import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

X, Y, test_x, test_y = mnist.load_data(one_hot=True)
print(X.shape, Y.shape, test_x.shape, test_y.shape)

print(X[0])
X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

print(type(X), type(Y), type(test_x), type(test_y))
print(X.shape, Y.shape, test_x.shape, test_y.shape)

# conv_net = input_data(shape=[None, 28, 28, 1], name='input')

# conv_net = conv_2d(conv_net, 32, 2, activation='relu')
# conv_net = max_pool_2d(conv_net, 2)

# conv_net = conv_2d(conv_net, 64, 2, activation='relu')
# conv_net = max_pool_2d(conv_net, 2)

# conv_net = fully_connected(conv_net, 1024, activation='relu')
# conv_net = dropout(conv_net, 0.8)

# conv_net = fully_connected(conv_net, 10, activation='softmax')
# conv_net = regression(conv_net, learning_rate=0.01, loss='categorical_crossentropy', name='targets')

# model = tflearn.DNN(conv_net)

# # model.fit(
# #     { 'input': X },
# #     { 'targets': Y },
# #     n_epoch=10,
# #     validation_set=({ 'input': test_x }, { 'targets': test_y }),
# #     snapshot_step=500,
# #     show_metric=True,
# #     run_id='mnist'
# # )
# # model.save('tflearncnn.model')

# model.load('tflearncnn.model')

# print(model.predict([test_x[1]]))