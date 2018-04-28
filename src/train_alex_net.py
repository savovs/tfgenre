import tflearn, os
from models import alex_net
from data import train_x, train_y, test_x, test_y, categories

# Create directories if necessary
save_path = os.path.dirname(os.path.realpath(__file__)) + '/../trained_models/alex_net'

if not os.path.exists(save_path):
    os.makedirs(save_path)

alex_model = tflearn.DNN(
    alex_net(len(categories)),
    tensorboard_verbose=3,
    tensorboard_dir='../logs',
)

alex_model.fit(
    { 'input': train_x },
    { 'targets': train_y },
    n_epoch=100,
    validation_set=({ 'input': test_x }, { 'targets': test_y }),
    snapshot_step=200,
    batch_size=64,
    show_metric=True,
    run_id='alex_net'
)

alex_model.save('../trained_models/alex_net/alex_net_model')
