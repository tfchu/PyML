'''
https://leemeng.tw/deep-learning-for-everyone-understand-neural-net-and-linear-algebra.html
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random

# https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
def rand_seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(0)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(0)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(0)

def load_axis():
    x = np.arange(-5, 5, 0.001)
    y0 = np.zeros((x.size, ))
    yp1 = np.zeros((x.size, )) + 1
    yn1 = np.zeros((x.size, )) - 1
    xy0 = np.concatenate((x.reshape((x.size, 1)), y0.reshape(x.size, 1)), axis=1)
    xyp1 = np.concatenate((x.reshape((x.size, 1)), yp1.reshape(x.size, 1)), axis=1)
    xyn1 = np.concatenate((x.reshape((x.size, 1)), yn1.reshape(x.size, 1)), axis=1)

    y = np.arange(-1, 1, 1/5000)
    x0 = np.zeros((y.size, ))
    xp1 = np.zeros((y.size, )) + 1
    xn1 = np.zeros((y.size, )) - 1
    x0y = np.concatenate((x0.reshape((x.size, 1)), y.reshape(x.size, 1)), axis=1)
    xp1y = np.concatenate((xp1.reshape((x.size, 1)), y.reshape(x.size, 1)), axis=1)
    xn1y = np.concatenate((xn1.reshape((x.size, 1)), y.reshape(x.size, 1)), axis=1)

    return xy0, xyp1, xyn1, x0y, xp1y, xn1y

def load_data(offset=1):
    x = np.arange(-5, 5, 0.001)
    y1 = -1 / (x**2 + 1)
    y2 = -1 / (x**2 + 1) + offset

    plt.plot(x, y1, 'o')
    plt.plot(x, y2, 'x')
    plt.grid()
    plt.show()

    # shuffled = np.random.permutation(range(x.size*2))

    x1_train = np.concatenate((x.reshape((x.size, 1)), y1.reshape(x.size, 1)), axis=1)
    x2_train = np.concatenate((x.reshape((x.size, 1)), y2.reshape(x.size, 1)), axis=1)
    x_train = np.concatenate((x1_train, x2_train))
    print(x1_train.shape)           # (10000, 2)
    print(x2_train.shape)           # (10000, 2)
    print(x_train.shape)            # (20000, 2)

    y1_train = np.zeros((x.size, )).reshape(x.size, 1)
    y2_train = np.ones((x.size, )).reshape(x.size, 1)
    y_train = np.concatenate((y1_train, y2_train))
    print(y1_train.shape)           # (10000, 1)
    print(y2_train.shape)           # (10000, 1)
    print(y_train.shape)            # (20000, 1)

    return x1_train, x2_train, x_train, y_train

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 1)                 2
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
'''
def one_layer_dnn():
    x1_train, x2_train, x_train, y_train = load_data()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
            1, 
            #use_bias=False, 
            input_shape=(2, )
        )
    )
    model.summary()

    model.compile(optimizer='adam',
                #loss='sparse_categorical_crossentropy',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=32)

    # weight and bias
    print(model.get_layer('dense').get_weights()[0])    # (2, 1)
    print(model.get_layer('dense').get_weights()[1])    # (1, )

    print(np.mean(model.predict(x1_train)))
    print(np.mean(model.predict(x2_train)))
    size = int(x1_train.shape[0])
    plt.plot(model.predict(x1_train).reshape((size,)), np.zeros(size,),'o')
    plt.plot(model.predict(x2_train).reshape((size,)), np.zeros(size,),'x')
    plt.grid()
    plt.show()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 2)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 3
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0
'''
def two_layer_dnn():
    x1_train, x2_train, x_train, y_train = load_data(0.5)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(
            2, 
            #use_bias=False, 
            input_shape=(2, ),
            activation='relu',
            kernel_initializer='he_normal',
            # bias_initializer='random_normal'
        )
    )
    model.add(tf.keras.layers.Dense(
            1, 
            #use_bias=False, 
            activation='relu',
            #kernel_initializer='he_normal',
            # bias_initializer='random_normal'
        )
    )
    model.summary()

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=100,
    #     decay_rate=0.9
    # )
    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    # opt = tf.keras.optimizers.SGD(
    #     learning_rate=lr_schedule, momentum=0.8
    # )
    model.compile(optimizer=opt,
                #loss='sparse_categorical_crossentropy',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=8)

    # weight and bias
    print(model.get_layer('dense').get_weights()[0])    # (2, 2)
    print(model.get_layer('dense').get_weights()[1])    # (2, )
    print(model.get_layer('dense_1').get_weights()[0])    # (2, 1)
    print(model.get_layer('dense_1').get_weights()[1])    # (1, )

    print(np.mean(model.predict(x1_train)))
    print(np.mean(model.predict(x2_train)))
    size = int(x1_train.shape[0])
    plt.plot(model.predict(x1_train).reshape((size,)), np.zeros(size,),'o')
    plt.plot(model.predict(x2_train).reshape((size,)), np.zeros(size,),'x')
    plt.grid()
    plt.show()

    layer_name = 'dense'
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output1 = intermediate_layer_model.predict(x1_train)
    intermediate_output2 = intermediate_layer_model.predict(x2_train)

    xy0, xyp1, xyn2, x0y, xp1y, xn1y = load_axis()
    intermediate_xy0 = intermediate_layer_model.predict(xy0)
    intermediate_xyp1 = intermediate_layer_model.predict(xyp1)
    intermediate_xyn2 = intermediate_layer_model.predict(xyn2)
    intermediate_x0y = intermediate_layer_model.predict(x0y)
    intermediate_xp1y = intermediate_layer_model.predict(xp1y)
    intermediate_xn1y = intermediate_layer_model.predict(xn1y)

    # print(intermediate_output1.shape)        # (10000, 2)
    plt.plot(intermediate_output1[:,0], intermediate_output1[:,1], 'o')
    plt.plot(intermediate_output2[:,0], intermediate_output2[:,1], 'x')
    plt.plot(intermediate_xy0[:,0], intermediate_xy0[:,1], '-', color='gray')
    plt.plot(intermediate_xyp1[:,0], intermediate_xyp1[:,1], '-', color='gray')
    plt.plot(intermediate_xyn2[:,0], intermediate_xyn2[:,1], '-', color='gray')
    plt.plot(intermediate_x0y[:,0], intermediate_x0y[:,1], '-', color='gray')
    plt.plot(intermediate_xp1y[:,0], intermediate_xp1y[:,1], '-', color='gray')
    plt.plot(intermediate_xn1y[:,0], intermediate_xn1y[:,1], '-', color='gray')
    plt.show()

if __name__ == '__main__':
    rand_seed()
    #one_layer_dnn()
    two_layer_dnn()