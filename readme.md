### A sample code base to execute an image classification pipeline using Convolutional D-CNN

#### Note: this is not aimed to acheive best performance on CatsDogs dataset; rather this is aimed
to build a image classification pipeline that is repeatable on any dataset. hope is that with minor
changes this can be adapted to any classification pipeline

#### CatsDogs data

Downloaded from [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765). The dataset
is downloaded and converted into TFRecords (this is added to the repo and managed by Git LFS).

#### Libraries

Tensorflow, Scikit

#### Environment

Python, Anaconda, GPU

#### Experiment

The `train.py` in `tf_main` directory contains code to run classification experiment. The data is batched using
TF `shuffle_batch` from TFRecords. Checkpoints are saved periodically using standard TF library. Threading/Pre-fetching
is handled within Tensorflow.

##### config.py

Single point to change any parameter.

##### network_def.py

The definition of networks.

#### Highlights

Following the CIFAR-10 example from [TFCifar10](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10),
this used multiple GPUs. Hhowever, it has not been tested with multiple GPUs. An interesting idea would be to write
the distribution of parameters accross multiple devices using a simple config file (perhaps in json/xml format).

NOTE : requires git-lfs.




