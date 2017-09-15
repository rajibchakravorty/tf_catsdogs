

import tensorflow as tf

import config

import sys
sys.path.append( '../tf_utility' )

import tf_helper_func as tf_helper

def get_logits( images ):

    with tf.variable_scope( 'conv_1' ) as scope:

        conv_kernel = tf_helper.variable_with_weight_decay( \
                        'weights', shape = [5,5,3,64],\
                        stddev = 5e-2, \
                        wd = 0.0, \
                        dtype_spec = tf.float32 )
        conv = tf.nn.conv2d( images, conv_kernel, [1,1,1,1], padding = 'SAME' )

        biases = tf_helper.variable_on_cpu( 'biases',\
                                            [64], \
                                            tf.constant_initializer( 0.001),\
                                            tf.float32 )

        pre_activation = tf.nn.bias_add( conv, biases )

        conv1 = tf.nn.relu( pre_activation )

        tf_helper.activation_summary( conv1, scope.name )

    pool1 = tf.nn.max_pool( conv1, ksize = [1,3,3,1], strides = [1,2,2,1], \
                            padding = 'SAME', name = 'pool1')

    norm1 = tf.nn.lrn( pool1, 4, bias=1.0, alpha = 0.001/9.0, beta = 0.75, \
                       name = 'norm1' )
     # local3
    with tf.variable_scope('local1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(norm1, [config.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf_helper.variable_with_weight_decay('weights', shape=[dim, 384],\
                                              stddev=0.04, wd=0.004, dtype_spec = tf.float32)
        biases = tf_helper.variable_on_cpu('biases', [384], tf.constant_initializer(0.1), tf.float32)
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='relu1')
        tf_helper.activation_summary(local1, scope.name)

    # local4
    with tf.variable_scope('local2') as scope:
        weights = tf_helper.variable_with_weight_decay('weights', shape=[384, 192],\
                                              stddev=0.04, wd=0.004, dtype_spec = tf.float32)
        biases = tf_helper.variable_on_cpu('biases', [192], tf.constant_initializer(0.1), tf.float32 )
        local2 = tf.nn.relu(tf.matmul(local1, weights) + biases, name='relu')
        tf_helper.activation_summary(local2, scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf_helper.variable_with_weight_decay('weights', [192, config.output_class],\
                                              stddev=1/192.0, wd=0.0, dtype_spec = tf.float32)
        biases = tf_helper.variable_on_cpu('biases', [config.output_class],\
                                  tf.constant_initializer(0.0), dtype_spec= tf.float32)
        logits = tf.add(tf.matmul(local2, weights), biases, name= 'softmax')
        tf_helper.activation_summary(logits, scope.name)

    return logits