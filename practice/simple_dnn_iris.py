'''
4 samples, each with 4 features. 1 label for each sample
 |
W0 (4x8)
 |
1 hidden-layer of 8 neurons
 |
W1 (8x1)
 |
1 output
'''
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_d(x):
    return x*(1-x)

def get_loss(y_cap, y):
    return np.sum(np.square(y_cap - y)) / 2

def get_output(X, W0, W1):
    return sigmoid(sigmoid(X.dot(W0)).dot(W1)).item(0)

eta = 0.5
iterations = 1000
X = np.array((
                (5.1, 3.5, 1.4, 0.2),
                (4.9, 3.0, 1.4, 0.2), 
				(6.2, 3.4, 5.4, 2.3),
                (5.9, 3.0, 5.1, 1.8)
            ))                                  # 4x4 iris data set
Y_cap = np.array(((0,), (0,), (1,), (1,)))      # 4x1 iris data set label

np.random.seed(0)
W0 = np.random.rand(4, 8)                       # 4x8
W1 = np.random.rand(8, 1)                       # 8x1

Y_list = np.array(())
loss_list = []

for i in range(iterations):
    # forward pass
    # layer i and h
    in_h = X.dot(W0)                            # 4x8
    out_h = sigmoid(in_h)                       # 4x8
    # layer h and o
    in_o = out_h.dot(W1)                        # 4x1
    Y = sigmoid(in_o)                           # 4x1
    # store Y
    if i == 0:
        Y_list = np.array(((Y.flatten())))
    else:
        Y_list = np.vstack((Y_list, Y.flatten()))
    # store loss
    loss_list.append(get_loss(Y_cap, Y))

    # backward pass
    # layer o and h
    dd = (Y - Y_cap) * Y * (1 - Y)              # 4x1
    g1 = out_h.T.dot(dd)                        # 8x1
    W1 = W1 - eta * g1                          # 8x1
    # layer h and i
    g2 = X.T.dot(dd.dot(W1.T)*sigmoid_d(out_h)) # 4x8
    W0 = W0 - eta * g2                          # 4x8

# test data
X_test = np.array(((4.0, 3.0, 6.0, 1.0)))
print("prediction: {}".format(get_output(X_test, W0, W1)))

# plot
X = np.arange(0, iterations, 1)
y0, = plt.plot(X, Y_list[:,0], 'o')
y1, = plt.plot(X, Y_list[:,1], 'o')
y2, = plt.plot(X, Y_list[:,2], '-')
y3, = plt.plot(X, Y_list[:,3], '-')
l, = plt.plot(X, loss_list)
plt.title('Output (Y) and Loss')
plt.legend((y0, y1, y2, y3, l), ('y0', 'y1', 'y2', 'y3', 'loss'))
plt.xlabel('iteration')
plt.ylabel('y/loss')
plt.grid()
plt.show()