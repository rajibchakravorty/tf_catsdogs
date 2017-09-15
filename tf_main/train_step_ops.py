import numpy
import re

import tensorflow as tf

import config as config

from network_def import get_logits


def get_ave_grads( grads ):
    average_grads = []
    for grad_and_vars in zip( *grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads



def train_steps() :

    with tf.device( '/cpu:0' ):
        global_step = tf.contrib.framework.get_or_create_global_step()

        # learning rate setup.
        learning_rate = tf.train.exponential_decay( config.init_learning_rate, \
                                                    global_step, \
                                                    config.steps_to_decay_learning_rate, \
                                                    config.init_learning_rate, \
                                                    staircase = True)
        #optimizer setup
        optimizer = tf.train.GradientDescentOptimizer( learning_rate )

        with tf.device( '/gpu:1' ):
            image_placeholder = tf.placeholder( tf.float32, shape = ( config.batch_size, \
                                                                      config.image_height, \
                                                                      config.image_width, \
                                                                      config.image_channel ) )
            label_placeholder = tf.placeholder( tf.int64, shape = ( config.batch_size ,) )

        all_grads = []
        for i in range( len( config.devices ) ):

            #with tf.device( config.devices[i] ) as scope:
            with tf.device( config.devices[i] ) as scope:

                logits = get_logits( image_placeholder )
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits( labels = label_placeholder, \
                                                                   logits = logits, name = 'cat_entropy' )
                loss_mean = tf.reduce_mean( loss, name ='cat_entropy_mean')
                tf.add_to_collection( 'losses', loss_mean )

                losses = tf.get_collection( 'losses', scope = scope )
                total_loss = tf.add_n( losses, name = 'total_loss' )

                for l in losses + [total_loss ]:
                    tf.summary.scalar( l.op.name, l )

                tf.get_variable_scope().reuse_variables()

                grads = optimizer.compute_gradients( total_loss )

                all_grads.append( grads )

        ave_grads = get_ave_grads( all_grads )

        tf.summary.scalar( 'learning_rate', learning_rate )

        train_op = optimizer.apply_gradients( ave_grads )

        for v in tf.trainable_variables():
            tf.summary.histogram( v.op.name + '/hist', v  )
            #tf.summary.histogram( var.op.name, var )

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram( var.op.name + '/grads', grad )

        #tf.get_variable_scope().reuse_variables()

        #variable_averages = tf.train.ExponentialMovingAverage( config.moving_average_decay, \
        #                                                       global_step )
        #variable_averages_op =variable_averages.apply( tf.trainable_variables() )

        # train_op = tf.group( apply_gradient_op, variable_averages_op )
        #train_op = apply_gradient_op
        summary_op = tf.summary.merge_all()

    return image_placeholder, label_placeholder, train_op, summary_op
