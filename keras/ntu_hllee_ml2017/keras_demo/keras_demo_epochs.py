import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist        # mnist: Modified National Institute of Standards and Technology database
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

def load_data():  # categorical_crossentropy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # take first 10,000 images and reshape
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

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

def train_model(ep = 20):
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()

    model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
    # model.add(Dense(input_dim=28 * 28, units=500, activation='sigmoid'))
    # model.add(Dropout(0.5))

    # model.add(Dense(units=500, activation='relu'))
    # model.add(Dense(units=500, activation='sigmoid'))

    # model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    # set configurations
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # train model
    # increasing batch_size makes result poor. with GPU this runs faster due to parallel computing 
    model.fit(x_train, y_train, batch_size=100, epochs=ep)

    # save model
    # ca: 'categorical_crossentropy' + 'adam'
    # ms: 'mse' + SGD(lr=0.1)
    # cs: 'categorical_crossentropy' + SGD(lr=0.1)
    # ma: 'mse' + 'adam'
    model.save('models/model_ca_ep_{}.h5'.format(ep))

    # evaluate the model and output the accuracy
    # result_train = model.evaluate(x_train, y_train)
    # result_test = model.evaluate(x_test, y_test)
    # print('Train Acc:', result_train[1])
    # print('Test Acc:', result_test[1])

def load_trained_model(ep = 20):
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # load trained model
    model = load_model('models/model_ca_ep_{}.h5'.format(ep))

    # evaluate the model and output the accuracy
    # result_train = model.evaluate(x_train, y_train)
    # result_test = model.evaluate(x_test, y_test)
    # print('Train Acc:', result_train[1])
    # print('Test Acc:', result_test[1])

    return model.evaluate(x_train, y_train)[1], model.evaluate(x_test, y_test)[1]

if __name__ == '__main__':
    epochs = [10, 20, 30, 40, 50]
    
    # train model
    # for ep in epochs:
    #     train_model(ep)

    # load trained model
    training_set_eval = []
    testing_set_eval = []

    for ep in epochs:
        res = load_trained_model(ep)
        training_set_eval.append(res[0])
        testing_set_eval.append(res[1])

    x = epochs
    y1 = training_set_eval
    y2 = testing_set_eval
    plot1, = plt.plot(x, y1, '-')
    plot2, = plt.plot(x, y2, '--')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('mnist accuracy')
    plt.legend((plot1, plot2), ('Training set', 'Testing set'))
    plt.grid()
    plt.show()