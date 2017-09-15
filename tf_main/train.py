import os

import tensorflow as tf
from train_step_ops import train_steps

import numpy as np

import config as config
from data_reader import read_and_decode as read_decode

feature = { 'height':    tf.FixedLenFeature( [], tf.int64 ), \
            'width':     tf.FixedLenFeature([], tf.int64),\
            'channel':   tf.FixedLenFeature( [], tf.int64 ), \
            'label':     tf.FixedLenFeature([], tf.int64), \
            'image_raw': tf.VarLenFeature( tf.float32 )\
            }



def train():
    g = tf.Graph()

    with g.as_default():

        image_placeholder, label_placeholder, train_op, summary_op = train_steps()
        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        images, labels = read_decode( config.train_data_file, \
                                      config.image_height,\
                                      config.image_width,\
                                      config.image_channel, \
                                      config.batch_size, \
                                      config.prefetcher_queue_capacity, \
                                      feature)

        tf.logging.set_verbosity( tf.logging.INFO )
        session_config = tf.ConfigProto()
        session_config.allow_soft_placement = True
        session_config.gpu_options.allow_growth = True
        session = tf.Session(graph=g, config=session_config ) #, \
            #allow_gpu_growth=True ))

        saver = tf.train.Saver(tf.global_variables())

    ##saving and writing summaries
    summary_writer = tf.summary.FileWriter(config.checkpoint_dir, g)
    session.run(init_op)

    with session:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners()

        for i in range(config.max_batches):
            ims, lbls = session.run( [images, labels])

            print 'Batch: {0}, Images {1}: '.format(i, np.mean(ims))

            session.run(train_op, feed_dict={image_placeholder: ims, \
                                             label_placeholder: lbls})

            if i % config.store_interval == 0:
                _, summary_str = session.run([train_op, summary_op], feed_dict={image_placeholder: ims, \
                                                                                label_placeholder: lbls})

                summary_writer.add_summary(summary_str, i)

                checkpoint_path = os.path.join(config.checkpoint_dir, 'model.ckpt')
                saver.save(session, checkpoint_path, global_step=i)
        coord.request_stop()
        coord.join( threads )
if __name__ == '__main__':
    train()
