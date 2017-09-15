
import numpy as np

import sys
sys.path.append( '../tf_utility' )


from prefetch_data import PreFetcher

class DataLoader():

    def __init__(self, data_file, arr_indicator, session, \
                 input_size, output_size, batch_size,\
                 queue_capacity, shuffle_data = False ):

        self.data_file     = data_file
        self.arr_indicator = arr_indicator
        self.data_length   = None
        self.session       = session
        self.input_size    = input_size
        self.output_size   = output_size
        self.batch_size    = batch_size
        self.queue_capacity=queue_capacity
        self.shuffle_data  = shuffle_data

        self.prefetcher = self._get_prefetcher()

    def _get_prefetcher( self ):

        dt = np.load( self.data_file )[ self.arr_indicator ]
        self.data_length = 100
        pf = PreFetcher( self.input_size, [], self.batch_size, \
                      dt[0:100], self.queue_capacity, self.session,\
                      self.shuffle_data )

        return pf

    def run_fetcher(self):

        self.prefetcher.run_fetch()

