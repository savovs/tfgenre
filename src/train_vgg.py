import tflearn, os
from models import vgg
from data import train_x, train_y, test_x, test_y, categories

# Create directories if necessary
save_path = os.path.dirname(os.path.realpath(__file__)) + '/../trained_models/simple'

if not os.path.exists(save_path):
    os.makedirs(save_path)

vgg_model = tflearn.DNN(
    vgg(len(categories)),
    tensorboard_verbose=3,
    tensorboard_dir='../logs',
)

vgg_model.fit(
    { 'input': train_x },
    { 'targets': train_y },
    n_epoch=100,
    validation_set=({ 'input': test_x }, { 'targets': test_y }),
    batch_size=64,
    snapshot_step=200,
    show_metric=True,
    run_id='vgg'
)

vgg_model.save('../trained_models/vgg/vgg_model')
