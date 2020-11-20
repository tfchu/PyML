import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist        # mnist: Modified National Institute of Standards and Technology database
from keras.preprocessing.image import array_to_img
from keras import backend as K
import matplotlib.pyplot as plt

def load_data():  # categorical_crossentropy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # take first 10,000 images and reshape
    number = 10000
    x_train = x_train[0:number]                                 # shape: (10000, 28, 28)
    y_train = y_train[0:number]                                 # shape: (10000, )

    # modify for CNN
    global input_shape
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(number, 1, 28, 28)            # shape: (10000, 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:                                                       # default
        x_train = x_train.reshape(number, 28, 28, 1)            # shape: (10000, 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)        

    # convert image array to float (from integer provided by mnist)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    # x_train = x_train
    # x_test = x_test

    x_test = np.random.normal(x_test)  # add noise

    # normalize the pixel values, now each value is 0 ~ 1
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 25)        250
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 25)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 50)        11300
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 50)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1250)              0
_________________________________________________________________
dense_1 (Dense)              (None, 500)               625500
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5010
=================================================================
Total params: 642,060
Trainable params: 642,060
Non-trainable params: 0
'''
def train_model():
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()

    # CNN
    '''
    when using data_format='channels_first' with input_shape=(1, 28, 28)
    get AttributeError: module 'tensorflow_core._api.v2.config' has no attribute 'experimental_list_devices'
    '''
    model.add(Conv2D(25, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    # model.add(Conv2D(filters=25, kernel_size=(3, 3), input_shape=(1, 28, 28)))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Conv2D(50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    model.add(Flatten())

    # full-connected network
    model.add(Dense(units=500, activation='relu'))
    # model.add(Dense(input_dim=28 * 28, units=500, activation='sigmoid'))
    # model.add(Dropout(0.5))

    # model.add(Dense(units=500, activation='relu'))
    # model.add(Dense(units=500, activation='sigmoid'))

    # model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    # model summary
    model.summary()
    # print(model.layers[0].get_weights())        # weights of filter

    # set configurations
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train model
    # increasing batch_size makes result poor. with GPU this runs faster due to parallel computing 
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    # save model
    # model.save('models/model_ca_cnn.h5')

    # evaluate the model and output the accuracy
    result_train = model.evaluate(x_train, y_train)
    result_test = model.evaluate(x_test, y_test)
    print('Train Acc:', result_train[1])        # 1.0
    print('Test Acc:', result_test[1])          # 0.984499990940094

if __name__ == '__main__':
    train_model()