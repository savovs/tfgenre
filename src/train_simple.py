import tflearn, os
from models import simple
from data import train_x, train_y, test_x, test_y, categories

# Create directories if necessary
save_path = os.path.dirname(os.path.realpath(__file__)) + '/../trained_models/simple'

if not os.path.exists(save_path):
    os.makedirs(save_path)

simple_model = tflearn.DNN(
    simple(len(categories)),
    tensorboard_verbose=3,
    tensorboard_dir='../logs',
)

simple_model.fit(
    { 'input': train_x },
    { 'targets': train_y },
    n_epoch=100,
    validation_set=({ 'input': test_x }, { 'targets': test_y }),
    batch_size=64,
    snapshot_step=200,
    show_metric=True,
    run_id='simple'
)

simple_model.save('../trained_models/simple/simple_model')

