#!/usr/bin/env python


import sys, os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/source/')


import source.classification  as cls




'''
The following function will be called to train and test your model.
The function name, signature and output type is fixed.
The first argument is file name that contain data for training.
The second argument is file name that contain data for test.
The function must return predicted values or emotion for each data in test dataset
sequentially in a list.
['sad', 'happy', 'fear', 'fear', ... , 'happy']
'''




def  aithon_level2_api(traingcsv, testcsv):

    # The following dummy code for demonstration.

    # Train the model with training data
    labelMap = cls.train_a_model(traingcsv)
     # Test that model with test data
    y_pred = cls.test_the_model(testcsv)
    y_pred_list = y_pred.astype('str').tolist()
    out_list = [''.join(row) for row in y_pred_list]
    res = [labelMap[label] for label in out_list]   
    # And return predicted emotions in a list
    return res





