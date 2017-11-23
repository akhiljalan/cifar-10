#!/usr/local/bin/python

#useful sources
#http://parneetk.github.io/blog/cnn-cifar10/

import pickle
import numpy as np 

def unpickle(file):
	'''
	file: A file (for our purposes, the cifar data batches)
	returns: A dictionary containing file data

	Not my script. Source: https://www.cs.toronto.edu/~kriz/cifar.html
	'''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_training_data():
	'''
	Returns a stacked version of all of the vectorized training data. 
	Each batch object is a dictionary. 
	Each batch dictionary is a 10,000 x 3,072 array. The row entries correspond
	to the RGB values of 32 x 32 images (32 x 32 x 3 = 3,072).
	Labels are integers corresponding to one of 10 possible image classes. 
	'''
	batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
	batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
	batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
	batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
	batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
	train_batches = [batch_1, batch_2, batch_3, batch_4, batch_5]

	all_training_features = np.vstack((batch_1[b'data'], batch_2[b'data'], 
                               batch_3[b'data'], batch_4[b'data'], batch_5[b'data']))

	all_training_labels = np.hstack((batch_1[b'labels'], batch_2[b'labels'], 
                               batch_3[b'labels'], batch_4[b'labels'], batch_5[b'labels']))

	all_training_features = all_training_features/255 

	return all_training_features, all_training_labels

def get_test_data(): 
	'''
	Returns the test data, which is in the same format as the training data. 
	test_features: The vectorized RGB values of each 32 x 32 images.
	test_labels: A vector integers corresponding to one of 10 possible image classes. 
	'''
	test_batch = unpickle('cifar-10-batches-py/test_batch')
	test_features = test_batch[b'data']
	test_labels = test_batch[b'labels']

	test_features = test_features/255

	return test_features, test_labels

def get_meta_data(): 
	'''
	meta_data: A dictionary of meta_data attributes, such as label string 
	names corresponding to each integer label. 
	'''
	meta_data = unpickle('cifar-10-batches-py/batches.meta')
	return meta_data