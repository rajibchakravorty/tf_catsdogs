
train_data_file = '/home/rachakra/catsdogs/prepare_data/train.tfrecords'
valid_data_file = '/home/rachakra/catsdogs/prepare_data/valid.tfrecords'
test_data_file  = '/home/rachakra/catsdogs/prepare_data/test.tfrecords'


image_height = 128
image_width  = 128
image_channel = 3
image_size = [image_height, image_width, image_channel]
output_class = 2

prefetcher_queue_capacity = 500

batch_size = 64

devices = ['/gpu:0']

tower_name = ['rachakra_0']


checkpoint_dir = 'checkpoints'


max_batches = 100000
store_interval = 1000

init_learning_rate = 0.1
steps_to_decay_learning_rate = 50000
decay_rate = 0.1


moving_average_decay = 0.9999
