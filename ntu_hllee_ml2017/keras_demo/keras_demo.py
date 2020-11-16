import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.datasets import mnist        # mnist: Modified National Institute of Standards and Technology database
from keras.preprocessing.image import array_to_img
from keras.layers import Input
import matplotlib.pyplot as plt
from keras import backend as K
from PIL import Image
import os

BATCH_SIZE = 100
EPOCHS = 20

def load_data():  # categorical_crossentropy
    # mnist has 60,000 training data, 10,000 test data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ''' tc: get x, y shape
    x: 3D array representing 60,000 images, each with 28x28 pixels, i.e. shape (60000, 28, 28)
    x_train: 
                    image 1                             image 60000
        ---------------|---------------       ---------------|---------------
    [   [[0 0 ... 0], ..., [0 0 ... 0]], ..., [[0 0 ... 0], ..., [0 0 ... 0]]   ]
    1   23: (60000, 28, 28)
    where 1: 60000 images, 2: 28-pixel wide, 3: 28-pixel high
    
    y: 1D array representing the number of each image
    y_train: [5 0 4 1 ... 5 6 8]
    '''
    # print(x_train.shape)    # (60000, 28, 28)   <- 3D array
    # print(y_train.shape)    # (60000, )         <- 1D array
    # print(x_test.shape)     # (10000, 28, 28)   <- 3D array
    # print(y_test.shape)     # (10000, )         <- 1D array

    ''' tc: show n-th picture
    an image is a tensor of shape (width, height, channels)
    grayscale image has 1 channel
    colorful image usually has 3 channels: R, G, B
    x_train is a vector of images
        x_train[0] is the 1st image of shape (28, 28), use np.expan_dims() to add image channel info
    '''
    # print(x_train[0].shape)                                 # (28, 28)        2D
    
    # show image with keras's array_to_img(), need adding 1 more dimension
    # img = array_to_img(np.expand_dims(x_train[0], axis=2))  # convert to (28, 28, 1)     3D
    # print(img.mode)             # L
    # print(img.size)             # (28, 28)
    # img.show()

    # show image with PIL fromarray()
    # img = Image.fromarray(x_train[0])
    # print(img.mode)             # L: based on training data shape, PIL knows it is grayscale
    # print(img.size)             # (28, 28)   
    # img.show()

    # take first 10,000 images and reshape
    number = 10000                                              # take first 10,000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)                  # convert shape (10,000, 28, 28) 3D to (10,000, 784) 2D array, i.e. 10000 x a vector of 784 elements
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)           # convert shape (10,000, 28, 28) 3D to (10,000, 784) 2D array

    # convert image array to float (from integer provided by mnist)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    ''' tc: convert 1D array to category-like (n classes) 2D array
    original y: [5 0 4 ...]
                           5                                0
            _______________|_______________  _______________|_______________
    now y: [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.], [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.], ...]
    '''
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    # x_train = x_train
    # x_test = x_test

    ''' tc: show image before and after adding noise, no visual difference
    np.random.normal(loc=0.0, scale=1.0, size=None) where loc is mean, scale is standard deviation, size is output shape
    now scale, size are not given, normal() return a single value with mean = x_test, values from many normal() calls have normal distribution
    '''
    # img = array_to_img(np.expand_dims(x_test[0].reshape(28, 28), axis=2))
    # img.show()
    x_test = np.random.normal(x_test)  # add noise
    # img = array_to_img(np.expand_dims(x_test[0].reshape(28, 28), axis=2))
    # img.show()
    
    ''' tc: grayscale byte image, value is 0 (black) ~ 255 (2^8 - 1) (white), also possible 0 (black) and 1 (white)
    normalize the pixel values, now each value is 0 ~ 1
    '''
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)

'''
tchu: input (784 input) -> dense_1 (500 neurons) -> dense_2 (output) (10 output)
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 500)               392500
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5010
=================================================================
Total params: 397,510
Trainable params: 397,510
Non-trainable params: 0
'''
def train_model():
    '''
    注意事项如下：
    1、batch_size=100,epochs=20为宜，batch_size过大会导致loss下降曲线过于平滑而卡在local minima、saddle point或plateau处，batch_size过小会导致update次数过多，运算量太大，速度缓慢，但可以带来一定程度的准确率提高
    2、hidden layer数量不要太多，不然可能会发生vanishing gradient(梯度消失)，一般两到三层为宜
    3、如果layer数量太多，则千万不要使用sigmoid等缩减input影响的激活函数，应当选择ReLU、Maxout等近似线性的activation function(layer数量不多也应该选这两个)
    4、每一个hidden layer所包含的neuron数量，五六百为宜
    5、对于分类问题，loss function一定要使用cross entropy(categorical_crossentropy)，而不是mean square error(mse)
    6、优化器optimizer一般选择adam，它综合了RMSProp和Momentum，同时考虑了过去的gradient、现在的gradient，以及上一次的惯性
    7、如果testing data上准确率很低，training data上准确率比较高，可以考虑使用dropout，Keras的使用方式是在每一层hidden layer的后面加上一句model.add(Dropout(0.5))，其中0.5这个参数你自己定；注意，加了dropout之后在training set上的准确率会降低，但是在testing set上的准确率会提高，这是正常的
    8、如果input是图片的pixel，注意对灰度值进行归一化，即除以255，使之处于0～1之间
    9、最后的output最好同时输出在training set和testing set上的准确率，以便于对症下药
    '''
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()

    model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
    # model.add(Dense(input_dim=28 * 28, units=500, activation='sigmoid'))
    # model.add(Dropout(0.5))

    model.add(Dense(units=500, activation='relu'))
    # model.add(Dense(units=500, activation='sigmoid'))

    # model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    # set configurations
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # train model
    # increasing batch_size makes result poor. with GPU this runs faster due to parallel computing 
    print('batch_size: {}, epochs: {}'.format(BATCH_SIZE, EPOCHS))
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save('models/keras_demo.h5')

    # evaluate the model and output the accuracy
    result_train = model.evaluate(x_train, y_train)
    result_test = model.evaluate(x_test, y_test)
    print('Train Acc:', result_train[1])        # 1.0
    print('Test Acc:', result_test[1])          # 0.9643999934196472

    # y = [[8.2438001e-13 8.2420079e-13 1.5844097e-12 1.3099312e-03 4.1797982e-19
    # 9.9869007e-01 1.2503906e-13 8.1099217e-14 8.3717045e-13 6.3655602e-11]]
    y = model.predict(x_train[0].reshape(1, 28*28))
    print(np.argmax(y, axis=None, out=None))            # 5: reverse of to_categorical()
    # y1 = model.predict(x_train[1])
    # y2 = model.predict(x_train[2])
    # print('x_train_0: {}, x_train_1: {}, x_train_2: {}'.format(y0, y1, y2))     # 


'''
Model: "mnist_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 784)               0
_________________________________________________________________
dense_1 (Dense)              (None, 500)               392500
_________________________________________________________________
dense_2 (Dense)              (None, 500)               250500
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5010
=================================================================
Total params: 648,010
Trainable params: 648,010
Non-trainable params: 0
'''
def train_model_functional_api():
    # define model
    inputs = Input(shape=(28*28, ))                 # input
    print(inputs.shape)                             # (None, 784)
    print(inputs.dtype)                             # <dtype: 'float32'>
    dense = Dense(500, activation="relu")           # 1st hidden 
    x = dense(inputs)
    x = Dense(500, activation="relu")(x)            # 2nd hidden
    outputs = Dense(10, activation="softmax")(x)    # output
    model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

    # train model
    (x_train, y_train), (x_test, y_test) = load_data()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=64, epochs=20)     # , validation_split=0.2
    
    # evaluate model
    test_scores_train = model.evaluate(x_train, y_train, verbose=2)
    print("Train loss:", test_scores_train[0])                          # 7.935219111677725e-05
    print("Train accuracy:", test_scores_train[1])                      # 1.0
    test_scores_test = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores_test[0])                            # 0.17261570996351647
    print("Test accuracy:", test_scores_test[1])                        # 0.9646999835968018

# layer weight/bias: [0] weight [1] bias
def get_weights_biases():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = load_model('models/keras_demo.h5')

    # input layer
    print('*** Input')
    print('Input img shape: {}'.format(x_train[0].shape))   # (784,)
    print(model.input.shape)                            # (None, 784)
    print(model.layers[0].input.shape)                  # (None, 784)
    # print(model.input)                                  # Tensor("dense_1_input:0", shape=(None, 784), dtype=float32)
    # print(model.inputs)                                 # [<tf.Tensor 'dense_1_input:0' shape=(None, 784) dtype=float32>]
    # print(model.inputs[0])                              # Tensor("dense_1_input:0", shape=(None, 784), dtype=float32)
    
    # 1st hidden layer w/b
    print('1st hidden layer weights/biases')
    print(model.get_layer('dense_1').get_weights()[0].shape)    # (784, 500)
    print(model.get_layer('dense_1').get_weights()[1].shape)    # (500, )
    print(model.layers[0].get_weights()[0].shape)       # (784, 500)
    print(model.layers[0].get_weights()[1].shape)       # (500, )

    # 1st layer output
    print('*** 1st hidden layer output')
    print(model.layers[0].output.shape)                 # (None, 500)
    print(model.layers[1].input.shape)                  # (None, 500)

    # output layer w/b
    print('Output layer weights/biases')
    print(model.get_layer('dense_2').get_weights()[0].shape)    # (500, 10)
    print(model.get_layer('dense_2').get_weights()[1].shape)    # (10, )
    print(model.layers[1].get_weights()[0].shape)       # (500, 10)
    print(model.layers[1].get_weights()[1].shape)       # (10, )

    # output layer output
    print('*** Output')
    print('output shape: {}'.format(y_train[0].shape))  # (10,)
    print(model.output.shape)                           # (None, 10)
    print(model.layers[1].output.shape)                 # (None, 10)

# get output of a certain layer
def get_layer_output():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = load_model('models/keras_demo.h5')

    # layer_name = 'dense_1'
    layer_name = 'dense_2'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(x_train[0].reshape((1, 784)))
    print(intermediate_output)

# alternative approach to get output of a certain layer
'''
K.function: 
'''
def get_layer_output1():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = load_model('models/keras_demo.h5')
    
    # layer_name = 'dense_1'
    layer_name = 'dense_2'
    # get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[3].output])
    get_output = K.function([model.input], [model.get_layer(layer_name).output])

    ''' why [0]? get only output
    [array([[3.8919418e-10, 1.9466567e-10, 3.4494394e-07, 1.5499649e-02,
        2.0365347e-16, 9.8450005e-01, 4.2405949e-12, 1.0950718e-08,
        2.9124286e-10, 4.3804227e-09]], dtype=float32)]
    '''
    layer_output = get_output([x_train[0].reshape((1, 784))])[0]
    print(layer_output)

# get final model output (output of output layer)
# sample output
# use model.output: [[3.8919418e-10 1.9466567e-10 ...]]
# use [model.output]: [array([[3.8919418e-10, ...]], dtype=float32)]
def get_model_output():
    (x_train, y_train), (x_test, y_test) = load_data()      # original training and test data
    model = load_model('models/keras_demo.h5')              # load trained model
    functor = K.function([model.input], model.output)       # function object to take model input and generate model output

    test = x_train[0].reshape((1, 784))                     # x_train[0]: handwriting 5 as test data
    print(functor([test]))                                  # run the function and take input to get output


# get output of all layers (including output layer) given input "test"
# without K.learning_phase() (bool: 0 or 1) also ok
def get_layer_outputs():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = load_model('models/keras_demo.h5')

    # layer output
    outputs = [layer.output for layer in model.layers]              # a list of all layer outputs
    functor = K.function([model.input, K.learning_phase()], outputs)

    # test = np.random.random([784])[np.newaxis,...]              # random input shape: (1, 784)
    test = x_train[0].reshape((1, 784))                             # x_train[0] as input
    layer_outs = functor([test, 0])
    print(layer_outs)

if __name__ == '__main__':
    # load_data()
    #train_model()
    train_model_functional_api()
    # get_weights_biases()
    # get_layer_output()
    # get_layer_output1()
    # get_model_output()
    # get_layer_outputs()