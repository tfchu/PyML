'''
fizzbuss
print the numbers from 1 to 100
    except that if the number is divisible by 3 print "fizz"
    if it's divisible by 5 print "buzz"
    if it's divisible by 15 print "fizzbuzz".
'''
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

NUM_DIGITS = 10
''' 
accuracy
NH  BS  NE  train               test
500 10 100  1.0                 0.9900990128517151
500 10  20  0.6652221083641052  0.5643564462661743
'''
NUM_HIDDEN = 500    # neurons per layer
BATCH_SIZE = 10     # batch_size
NUM_EPOCHS = 100     # epochs

'''
x (features): binary encode of an integer
LSB goes first, e.g. 101 -> 00_0110_0101b -> [1 0 1 0 0 1 1 0 0 0]
'''
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

'''
y (label): [a, b, c, d]
[0, 0, 0, 1] if the number can be divided by 15 (either 3 or 5)
[0, 0, 1, 0] if the number can be divided by 5
[0, 1, 0, 0] if the number can be divided by 3
[1, 0, 0, 0] otherwise
'''
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

def load_data():
    x_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    y_train = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])
    x_test = np.array([binary_encode(i, NUM_DIGITS) for i in range(101)])
    y_test = np.array([fizz_buzz_encode(i)          for i in range(101)])
    return x_train, y_train, x_test, y_test

# keras
def main():
    x_train, y_train, x_test, y_test = load_data()

    model = Sequential()
    model.add(Dense(input_dim=NUM_DIGITS, units=NUM_HIDDEN, activation='relu'))        # input_dum: 10 binary values, units: neurons per layer
    # model.add(Dense(units=NUM_HIDDEN, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))                                     # units: 4 labels
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

    # evaluate the model and output the accuracy
    result_train = model.evaluate(x_train, y_train)
    result_test = model.evaluate(x_test, y_test)
    print('Train Acc:', result_train[1])            # 1.0               , 0.9956663250923157
    print('Test Acc:', result_test[1])              # 0.9801980257034302, 0.9207921028137207

# tensorflow
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def main_tf():
    trX, trY, teX, teY = load_data()

    X = tf.placeholder("float", [None, NUM_DIGITS])
    Y = tf.placeholder("float", [None, 4])

    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, 4])

    py_x = model(X, w_h, w_o)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    predict_op = tf.argmax(py_x, 1)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        for epoch in range(NUM_EPOCHS):
            p = np.random.permutation(range(len(trX)))
            trX, trY = trX[p], trY[p]

            for start in range(0, len(trX), BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            
            print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))
        
        numbers = np.arange(1, 101)

        teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
        teY = sess.run(predict_op, feed_dict={X: teX})
        output = np.vectorize(fizz_buzz)(numbers, teY)

        print(output)

if __name__ == '__main__':
    main_tf()