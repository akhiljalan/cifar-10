#!/usr/local/bin/python

from preprocessing import get_meta_data

def accuracy(prediction, true_labels): 
    '''
    prediction: A vector of predictions, whose values are in [1, 2, ..., 10]
    true_labels: A vector of true image labels, whose values are in [1, 2, ..., 10]
    
    returns: the accuracy rate of a prediction against the true labels.     
    
    Assumes both prediction and true_labels contain integers only, although
    '''
    assert len(prediction) == len(true_labels), 'Mismatched prediction and label set'
    prediction = np.int_(np.rint(np.array(prediction))) #round to nearest integer and cast to integer type

    num_accurate = 0
    for i in range(len(prediction)): 
        if(prediction[i] == true_labels[i]): 
            num_accurate += 1 
    return (num_accurate/len(prediction))


def number_to_name(num): 
    '''
    num: An integer label in [1, 2, ..., 10] 
    returns: the corresponding string which the label corresponds to. 

    number_to_name(3)
    >>> 'bird'
    '''
    label_names = meta_data[b'label_names']
    assert type(num) == int, '{} is not an integer'.format(num)
    assert num in [x for x in range(1, 11)], '{} is not between 1 and 10'.format(num)
    return label_names[num - 1].decode('utf-8')