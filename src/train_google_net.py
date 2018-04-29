import tflearn, os
from models import google_net
from data import train_x, train_y, test_x, test_y, categories

# Create directories if necessary
save_path = os.path.dirname(os.path.realpath(__file__)) + '/../trained_models/google_net'

if not os.path.exists(save_path):
    os.makedirs(save_path)

model = tflearn.DNN(
    google_net(len(categories)),
    tensorboard_verbose=3,
    tensorboard_dir='../logs',
)

model.fit(
    { 'input': train_x },
    { 'targets': train_y },
    n_epoch=100,
    validation_set=({ 'input': test_x }, { 'targets': test_y }),
    batch_size=64,
    snapshot_step=200,
    show_metric=True,
    shuffle=True,
    run_id='google_net'
)

model.save('../trained_models/google_net/google_net_model')

