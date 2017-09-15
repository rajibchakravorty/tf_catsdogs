from skimage.io import imsave

import tensorflow as tf

import numpy as np

import skimage.io as io




def read_and_decode( record_file, height, width, channel, batch_size, capacity, feature ):


    filename_queue = tf.train.string_input_producer([record_file] ) #, num_epochs=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read( filename_queue )

    fts = tf.parse_single_example( serialized_example, features = feature )


    image = tf.sparse_tensor_to_dense( fts[ 'image_raw' ], default_value = 0. )
    image = tf.reshape( image, [height, width, channel] )

    lbl   = tf.cast( fts['label'], tf.int64 )

    #height  = tf.cast( fts['height'], tf.int64 )
    #width   = tf.cast( fts['width'], tf.int64 )
    #channel = tf.cast( fts['channel'], tf.int64 )


    images, labels = tf.train.shuffle_batch( [image, lbl], batch_size = batch_size, capacity = capacity, \
                                             num_threads = 2, min_after_dequeue = 10 )

    #lbls, hs, ws, cs = tf.train.shuffle_batch([lbl, height, width, channel], batch_size=10, capacity=50, \
    #                                        num_threads=1, min_after_dequeue=20)


    return images, labels #, lbl
    #return lbls, hs, ws, cs

'''
def test_iterator( filename ):

    record_iterator = tf.python_io.tf_record_iterator( path = filename )

    count = 1
    for string_record in record_iterator:

        example = tf.train.Example()

        example.ParseFromString( string_record )

        img_list = ( example.features.feature['image_raw'] ).float_list.value[:]

        #print img_list

        img_array = np.array( img_list )

        #img_1d = np.fromstring( img_array, dtype = np.float32 )

        print img_array.shape

        img_2d = np.reshape( img_array, [128,128,3] )
        print img_2d.shape

        imsave( 'test_{0}.jpg'.format( count ), img_2d )

        count += 1

        #if count ==10:
        #    break

if __name__ == '__main__':

    test_iterator( '../prepare_data/test.tfrecords' )
'''
