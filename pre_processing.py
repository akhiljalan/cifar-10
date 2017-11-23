#!/usr/local/bin/python

#Script source: 
#https://www.cs.toronto.edu/~kriz/cifar.html

#useful sources
#http://parneetk.github.io/blog/cnn-cifar10/

import pickle
import numpy as np 

def unpickle(file):
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
	test_batch = unpickle('cifar-10-batches-py/test_batch')
	test_features = test_batch[b'data']
	test_labels = test_batch[b'labels']
	return test_features, test_labels

def get_meta_data(): 
	meta_data = unpickle('cifar-10-batches-py/batches.meta')





'''Standardize the image pixel values, which are normally in [0, 255].'''

test_features = test_features/255
all_training_features = all_training_features/255

'''Some useful functions.'''

label_names = meta_data[b'label_names']

'''Returns the accuracy rate of a prediction against the true labels.
Assumes both prediction and true_labels contain integers only, although'''
def accuracy(prediction, true_labels): 
    assert len(prediction) == len(true_labels), 'Mismatched prediction and label set'
    prediction = np.int_(np.rint(np.array(prediction))) #round to nearest integer and cast to integer type

    num_accurate = 0
    for i in range(len(prediction)): 
        if(prediction[i] == true_labels[i]): 
            num_accurate += 1 
    return (num_accurate/len(prediction))

'''Takes a label value (an integer between 1 and 10) and returns the corresponding
string which the label corresponds to. Example: 3 --> bird '''
def number_to_name(num): 
    assert type(num) == int, '{} is not an integer'.format(num)
    assert num in [x for x in range(1, 11)], '{} is not between 1 and 10'.format(num)
    return label_names[num - 1].decode('utf-8')