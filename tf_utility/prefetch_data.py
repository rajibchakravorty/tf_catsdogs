import time
import numpy as np

import threading

import tensorflow as tf


import sys
sys.path.append( '../ml_utility')
from image_object import ImageObject

class PreFetcher():

    def __init__(self, image_shape, label_shape, batch_size, \
                    source_data ,queue_capacity, tf_session, shuffle_data ):

        self.image_shape = image_shape
        self.label_shape = label_shape
        self.batch_size  = batch_size
        self.data        = source_data

        self._image_placeholder = tf.placeholder( tf.float32, \
                                        self.image_shape, name = 'input_image'  )
        self._label_placeholder = tf.placeholder( tf.int32, shape = self.label_shape,
                                                  name = 'input_label')

        self._tf_session = tf_session
        self.queue_capacity = queue_capacity

        self._shuffle_data = shuffle_data

        self._get_queue_ops()

        self.t = threading.Thread( target = self._fetch_data )
        self.t.daemon = True

    def _get_queue_ops(self):


        q = tf.FIFOQueue( self.queue_capacity, [tf.float32, tf.int32 ]  , \
                          shapes = [ self.image_shape,self.label_shape ] )

        self.enq_op = q.enqueue( [self._image_placeholder, self._label_placeholder] )

        #self.deq_op   = q.dequeue_many( [self.batch_size] )

        self.deq_op = q.dequeue_up_to(self.batch_size)

    def _fetch_data(self):

        while True:
            if self._shuffle_data:
                np.random.shuffle( self.data )
            for d in self.data:

                try:
                    image = d.get_image()
                    label = d.label
                    self._tf_session.run( self.enq_op, feed_dict = { self._image_placeholder: image, \
                                                   self._label_placeholder: label }  )


                except:
                    continue

    def run_fetch(self):

        self.t.start()

'''
if __name__ == '__main__':

    data_file = '../prepare_data/cats_dogs.npz'
    dt = np.load( data_file )['arr_2']

    print 'Data loaded, creating session'
    sess = tf.Session()

    print 'Creating Prefetcher'
    pf = PreFetcher( [128,128,3], [], 25, dt[0:100], 50, sess, True  )
    pf.run_fetch()


    for i in range( 20 ):

        print 'Looping ', i
        start_time = time.time()
        im, l = sess.run( pf.deq_op )
        print im.shape
        from skimage.io import imsave
        for j in range( im.shape[0] ):
            imsave( 'test/sample_{0}_{1}.jpg'.format( i,j ), im[j] )
            print time.time() - start_time

    #coord.request_stop()
    #coord.join( [t] )
'''