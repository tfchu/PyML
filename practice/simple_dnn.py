'''
network: 
2 input (i1 = 0.5, i2 = 0.1): 1 sample of 2 features
 |
weights: w1, w2, w3, w4
 |
1 hidden layer with 2 neurons (h1, h2)
 |
weights: w5, w6, w7, w8
 |
2 output (y1 =  0.1, y2 = 0.99)

notes: 
- activation function: all sigmoid
- loss function: mean-square error
- only weight, no bias
'''
import numpy as np
import matplotlib.pyplot as plt

i1, i2 = 0.5, 0.1
w1, w2, w3, w4, w5, w6, w7, w8 = 0.15, 0.2, 0.25, 0.30, 0.40, 0.45, 0.50, 0.55
y_cap1, y_cap2 = 0.1, 0.99
y_cap = np.array(((y_cap1, y_cap2)))
eta = 0.5
k1 = 0  # middle term for w5, w6
k2 = 0  # middle term for w7, w8
mse_list = []
iterations = 10000

# output of each layer
def get_layer_out(weight, input):
    return 1/(1+np.exp(-weight.dot(input)))

# MSE
def get_error(y_cap, y):
    return np.sum(np.square(y_cap - y)) / 2

# gradient before output layer: w5, w6, w7, w8
def get_gradient_output_layer():
    global k1, k2, out_h
    k1 = (y[0] - y_cap[0]) * y[0] * (1 - y[0])
    gradient1 = k1 * out_h
    k2 = (y[1] - y_cap[1]) * y[1] * (1 - y[1])
    gradient2 = k2 * out_h
    return np.concatenate((gradient1, gradient2), axis=0)

# gradient before 1st hidden layer counted backward: w1, w2, w3, w4
def get_gradient_hidden_layer_back1():
    gradient1 = (k1 * w5 + k2 * w7) * out_h[0] * (1 - out_h[0]) * input
    gradient2 = (k1 * w6 + k2 * w8) * out_h[1] * (1 - out_h[1]) * input
    return np.concatenate((gradient1, gradient2), axis=0)

# initial forward pass
input = np.array(((i1, i2)))
out_h = get_layer_out(np.array(((w1, w2), (w3, w4))), input)
y = get_layer_out(np.array(((w5, w6), (w7, w8))), out_h)
mse = get_error(y_cap, y)
print(0, w1, w2, w3, w4, w5, w6, w7, w8, mse)

# update weights
for i in range(iterations):
    # get gradients - go before updating weights as we need original weight values
    g_output_layer = get_gradient_output_layer()                # gradient before output layer
    g_hidden_layer_back1 = get_gradient_hidden_layer_back1()    # then back2, back3 ... if more layers (back-propagation)
    # update w5, w6, w7, w8
    o = np.array(((w5, w6, w7, w8))) - eta * g_output_layer
    w5, w6, w7, w8 = o[0], o[1], o[2], o[3]
    # update w1, w2, w3, w4
    o = np.array(((w1, w2, w3, w4))) - eta * g_hidden_layer_back1
    w1, w2, w3, w4 = o[0], o[1], o[2], o[3]
    # get output of 1st layer
    out_h = get_layer_out(np.array(((w1, w2), (w3, w4))), input)
    # get output
    y = get_layer_out(np.array(((w5, w6), (w7, w8))), out_h)
    # total loss
    mse = get_error(y_cap, y)
    mse_list.append(mse)
    # print updated weights
    if i % 1000 == 0:
        print(i+1, w1, w2, w3, w4, w5, w6, w7, w8, mse)

# loss, y_cap and y
print('final loss: {}'.format(mse))
print('actual y1: {}, y2: {}'.format(y_cap[0], y_cap[1]))
print('predicted y1: {}, y2: {}'.format(y[0], y[1]))
# plot loss
X = np.arange(0, iterations, 1)
plt.plot(X[-500:], mse_list[-500:])
plt.grid()
plt.show()
