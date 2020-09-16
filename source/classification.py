



import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128,(3,3),padding = "same",input_shape= (48,48,1),kernel_regularizer=l2(0.01)), #48
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64,(3,3),padding = "same",kernel_regularizer=l2(0.01)),#48
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),#24
        tf.keras.layers.Conv2D(64,(3,3),padding = "same",kernel_regularizer=l2(0.01)),#24
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),#12
        tf.keras.layers.Conv2D(128,(3,3),padding = "same",kernel_regularizer=l2(0.01)),#12
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128,(3,3),kernel_regularizer=l2(0.01)),#10
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128,(3,3),kernel_regularizer=l2(0.01)),#8
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),#4
        tf.keras.layers.Conv2D(512,(3,3),padding = "same",kernel_regularizer=l2(0.01)),#4
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(512,(3,3),kernel_regularizer=l2(0.01)),#2
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(1024,(1,1),kernel_regularizer=l2(0.01)),#2
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2,2),#1
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3,activation = tf.nn.softmax)
    ])
    model.compile(optimizer = Adagrad(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_a_model(trainfile):
    '''
    :param trainfile: csv file containing pixel values of the training images and labels in the "emotion" column
    :return: a map from labels to one-hot-encoders
    '''
    x = pd.read_csv(trainfile)
    y = x["emotion"]
    x_t = np.array(x.loc[:,"pixel_0":]).reshape((-1,48,48,1))
    labels = pd.get_dummies(y)
    keys = list(labels.columns)
    labelMap  = {'100':keys[0],'010':keys[1],'001':keys[2]}
    
    trainDatagen = ImageDataGenerator(
                    rescale = 1./255,
                    rotation_range= 5,
                    width_shift_range= 0.1,
                    height_shift_range= 0.1,
                    horizontal_flip= True,
                    fill_mode = 'nearest',
                    shear_range = 0.1,
                    validation_split = 0.1)
    
    trainGenerator = trainDatagen.flow(x_t,labels, batch_size=32, subset = "training", shuffle = True)
    validationGenerator = trainDatagen.flow(x_t,labels,batch_size = 32, subset = "validation")
    
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5", 
                                    monitor = 'val_accuracy',
                                    verbose=1, 
                                    save_best_only = True)
    
    model = build_model()
    model.fit(trainGenerator,
          steps_per_epoch  = trainGenerator.__len__(), 
          validation_steps = validationGenerator.__len__(),
          validation_data = validationGenerator,
          callbacks=[checkpointer],
          epochs = 150
         )
    
    return labelMap




def test_the_model(testfile):
    '''
    :param testfile: csv file containing pixel values of test images
    :return:  a list of predicted values in same order as the testfile
    '''
    
    h = pd.read_csv(testfile,header = None)    
    x_test = np.array(h).reshape((-1,48,48,1))/255
    model = build_model()
    model.load_weights("best_weights.hdf5")
    y_predicted = model.predict(x_test)
    temp = y_predicted.max(axis = 1,keepdims = True) == y_predicted
    y_predicted = temp.astype('int')

    return y_predicted



    
    
    
    
    
    
    
    
