from os import listdir
from os.path import isfile, join

import numpy as np

from random import random

import tensorflow as tf

import sys
sys.path.append( '../ml_utility' )

from image_object import ImageObject
from transformer import Transformer


image_dim = (128,128,3)


max_translation = 30
max_rotation    = 90.
do_flip         = True


num_label = 2 #cat = 0, dog =1


do_balance = True
augment_factor = 5

def prepare_original_imo_label( label_path, label ):

    image_files = [join(label_path, f ) for f in listdir( label_path ) \
                      if isfile( join( label_path, f ) ) ]

    imos = list()
    for imf in image_files:

        transform = Transformer( translation = (0,0 ),\
                                 rotation    = 0.0, \
                                 resize_shape = ( image_dim[0], image_dim[1] ),\
                                 do_flip = False,\
                                 convert_to_grey = False )

        imo = ImageObject( imf, label, transform )

        imos.append( imo )

    return imos

def prepare_original_imo( folder ):

    imos = list()

    count_labels = list()
    for label in range( num_label ):

        label_path = join( folder, '{0}'.format( label ) )
        imo_label  = prepare_original_imo_label( label_path, label )

        imos += imo_label

        count_labels.append( len( imo_label ) )

    return imos, count_labels

def balance_factor( count_labels ):

    count_labels = np.array( count_labels )

    max_idx = np.argsort( count_labels )

    max_count = count_labels[ max_idx[-1] ]

    factors = float( max_count ) /count_labels

    return factors

def augment( factor, original_imos ):

    augmented_imos = list()
    for imo in original_imos:

        for f in range( factor ):

            translation = ( int( ( random() - 0.5 ) * max_translation ),\
                        int( ( random() - 0.5 ) * max_translation ) )
            rotation = random() *max_rotation


            transform = Transformer( translation = translation,\
                                 rotation    = rotation, \
                                 resize_shape = ( image_dim[0], image_dim[1] ),\
                                 do_flip = do_flip,\
                                 convert_to_grey = False )

            aug_imo = ImageObject( imo.image_path , imo.label, transform )

            augmented_imos.append( aug_imo )

    return augmented_imos




def prepare_data( original_imos, count_labels, do_balance, aug_factor ):

    if do_balance == True:
        bfac = balance_factor( count_labels )

    additional_imos = list()
    start_idx = 0
    for l in range( num_label):

        end_idx = start_idx + count_labels[l]

        if do_balance == True:
            factor = int( bfac[l] * aug_factor )
        else:
            factor = aug_factor

        augmented_imos = augment( factor, original_imos[start_idx:end_idx] )

        start_idx = end_idx

        additional_imos += augmented_imos
    return additional_imos

def _float_feature( value ):

    return tf.train.Feature( float_list = tf.train.FloatList( value = value ) )

def _bytes_feature( value ):

    return tf.train.Feature( bytes_list = tf.train.BytesList( value = [value] ) )

def _int64_feature( value ):

    return tf.train.Feature( int64_list = tf.train.Int64List( value = [value] ) )

def save_to_tfrecord(output_file, input_list):

    writer =  tf.python_io.TFRecordWriter(output_file)

    for il in input_list:

        try:
            img = il.get_image()
            lbl = il.label
        except IOError:
            continue
        except ValueError:
            continue

       #print img.shape

        height = img.shape[0]
        width  = img.shape[1]

        channel = 3
        if img.shape[2] != 3: #get_image returns image in 3-D, if grey, the channel becomes 1
            print il.image_path
            continue

        #assert img.shape == (128,128,3)

        img_ravel = np.ravel( img )
        #print len( img_string )

        example = tf.train.Example( features = tf.train.Features( feature ={\
            'height' : _int64_feature( height ), \
            'width' : _int64_feature( width ), \
            'channel' : _int64_feature( channel), \
            'label' : _int64_feature( lbl ),
            'image_raw' : _float_feature( img_ravel )\
            } ))

        writer.write( example.SerializeToString() )


    writer.close()

def main():

    train_folder = '../data'

    train_orig_imos, count_labels = prepare_original_imo( train_folder )
    np.random.shuffle( train_orig_imos )

    num_samples = len( train_orig_imos )
    print 'No of images : {0}'.format( num_samples )
    training_samples = int( num_samples * 0.8 )
    valid_samples    = int( num_samples * 0.1 )


    train_list = train_orig_imos[0:training_samples]
    valid_list = train_orig_imos[training_samples:( training_samples + valid_samples)]
    test_list  = train_orig_imos[( training_samples + valid_samples):]

    train_aug_imos = prepare_data( train_list, count_labels, do_balance, augment_factor )
    print 'Train {0}, {1}'.format( len( train_orig_imos ), len( train_aug_imos ) )



    print 'Before augmentation : Train/Valid/Test : {0},{1},{2}'.format( len( train_list ), len( valid_list ),\
                                                   len( test_list ) )

    train_list += train_aug_imos

    print 'After augmentation : Train/Valid/Test : {0},{1},{2}'.format( len( train_list ), len( valid_list ),\
                                                   len( test_list ) )
    print 'Saving test records'
    save_to_tfrecord( 'test.tfrecords', test_list )
    print 'Saving valid records'
    save_to_tfrecord('valid.tfrecords', valid_list)
    print 'Saving train records'
    save_to_tfrecord( 'train.tfrecords', train_list)

    #data_file = 'cats_dogs.npz'

    #np.savez( data_file, train_list, valid_list, test_list )

if __name__ == '__main__':

    main()
