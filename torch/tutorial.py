# https://colab.research.google.com/drive/1Xed5YSpLsLfkn66OhhyNzr05VE89enng#scrollTo=vQBkFt9LMTiO
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(446)  # repeatable randomness
np.random.seed(446)     # repeatable randomness

# '''
# compare operations of numpy and torch
# '''
# # we create tensors in a similar way to numpy nd arrays
# x_numpy = np.array([0.1, 0.2, 0.3])
# x_torch = torch.tensor([0.1, 0.2, 0.3])
# print('x_numpy, x_torch')
# print(x_numpy, x_torch)
# print()

# # to and from numpy, pytorch
# print('to and from numpy and pytorch')
# print(torch.from_numpy(x_numpy), x_torch.numpy())
# print()

# # we can do basic operations like +-*/
# y_numpy = np.array([3,4,5.])
# y_torch = torch.tensor([3,4,5.])
# print("x+y")
# print(x_numpy + y_numpy, x_torch + y_torch)
# print()

# # many functions that are in numpy are also in pytorch
# print("norm")
# print(np.linalg.norm(x_numpy), torch.norm(x_torch))
# print()

# # to apply an operation along a dimension,
# # we use the dim keyword argument instead of axis
# '''
# 0-th dim: across rows (1, 3), (2, 4)
# 1-th dim: across cols (1, 2), (3, 4)
# [
#     [1, 2]
#     [3, 4]
# ]
# '''
# print("mean along the 0th dimension")
# x_numpy = np.array([[1,2],[3,4.]])
# x_torch = torch.tensor([[1,2],[3,4.]])
# print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))
# print(np.mean(x_numpy, axis=1), torch.mean(x_torch, dim=1))

# '''
# use view to reshape tensors
# '''
# # "MNIST"
# N, C, W, H = 10000, 3, 28, 28   # number of samples, channel, width, height
# X = torch.randn((N, C, W, H))

# print(X.shape)
# print(X.view(N, C, 784).shape)
# print(X.view(-1, C, 784).shape) # automatically choose the 0th dimension with -1, only one dim can be -1


# '''
# "broadcast" in torch
# PyTorch operations support NumPy Broadcasting Semantics.
# broadcast when dim does not exist, or dim = 1
# final dim: larger dim of all tensors
# original: reshape y to 1,3,1,1, and copy 5 times, then x + y
# x is 5 of these
# tensor([[[[0.],
#           [0.],
#           [0.],
#           [0.]]],
#         [[[0.],
#           [0.],
#           [0.],
#           [0.]]],
#         ...
# e.g. broadcast
# [[0, 0, 0, 0], + [[1, 2, 3, 4]] = [[1, 2, 3, 4],
#  [0, 0, 0, 0]]                     [1, 2, 3, 4]]
# '''
# x=torch.empty(5,1,4,1)
# y=torch.empty(  3,1,1)
# print((x+y).size()) # [5, 3, 4, 1]

# '''
# Computation graphs
# '''
# a = torch.tensor(2.0, requires_grad=True) # we set requires_grad=True to let PyTorch know to keep the graph
# b = torch.tensor(1.0, requires_grad=True)
# c = a + b
# d = b + 1
# e = c * d
# print('c', c)
# print('d', d)
# print('e', e)

# '''
# derivative of f(x) solved manually and by torch
# then use gradient descent to find x that minimizes f(x)
# '''
# def f(x):
#     return (x-2)**2

# def fp(x):
#     return 2*(x-2)

# x = torch.tensor([1.0], requires_grad=True)

# y = f(x)
# y.backward()        # then you know x.grad

# print('Analytical f\'(x):', fp(x))      # solve manually
# print('PyTorch\'s f\'(x):', x.grad)     # solve by torch

# # gradient descent to find x that minimized f(x), start at x = 1.0 (above)
# x = torch.tensor([5.0], requires_grad=True) # init value
# step_size = 0.1
# print('iter,\tx,\tf(x),\tf\'(x),\tf\'(x) pytorch')
# for i in range(25):
#     y = f(x)
#     y.backward()
#     print('{},\t{:.3f},\t{:.3f},\t{:.3f},\t{:.3f}'.format(i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))
#     x.data = x.data - step_size * x.grad
#     x.grad.detach_()
#     x.grad.zero_()

# '''
# w = [w0, w1]
# g(w) = 2*w0*w1 + w1*cos(w0)
# compute gradient of g(w), and set w = [2, pi - 1]
# '''
# def g(w):
#     return 2*w[0]*w[1] + w[1]*torch.cos(w[0])

# # [gradient of w0, gradient of w1]
# def grad_g(w):
#     return torch.tensor([2*w[1] - w[1]*torch.sin(w[0]), 2*w[0] + torch.cos(w[0])])

# w = torch.tensor([np.pi, 1], requires_grad=True)

# z = g(w)
# z.backward()

# print('Analytical grad g(w)', grad_g(w))    # solve manually
# print('PyTorch\'s grad g(w)', w.grad)       # solve by torch


# '''
# linear regression
# y = X * w + noise
# loss function: RSS (residual sum of squares)
#     https://en.wikipedia.org/wiki/Residual_sum_of_squares

# tensor.detach() creates a tensor that shares storage with tensor that does not require grad. then convert to numpy()
#     Tensor that requires grad cannot call numpy()
# '''
# d = 2       # 2 features
# n = 50      # 50 samples
# X = torch.randn(n,d)                        # inputs (samples)
# true_w = torch.tensor([[-1.0], [2.0]])      # actual weights
# y = X @ true_w + torch.randn(n,1) * 0.1     # labels with noise
# print('X shape', X.shape)
# print('y shape', y.shape)
# print('w shape', true_w.shape)

# # define a linear model with no bias
# def model(X, w):
#     return X @ w    # (50x2)x(2x1) = (50x1) or (samples x output)

# # the residual sum of squares loss function
# # y: labels
# # y_hat: outputs
# def rss(y, y_hat):
#     return torch.norm(y - y_hat)**2 / n

# # analytical expression for the gradient
# def grad_rss(X, y, w):
#     return -2*X.t() @ (y - X @ w) / n

# w = torch.tensor([[1.], [0]], requires_grad=True)   # start weights
# y_hat = model(X, w)

# loss = rss(y, y_hat)
# loss.backward()

# # sanity check only
# print('Analytical gradient', grad_rss(X, y, w).detach().view(2).numpy())
# print('PyTorch\'s gradient', w.grad.view(2).numpy())

# # gradient descent to find optimal wights
# step_size = 0.1
# print('iter,\tloss,\tw')
# for i in range(20):
#     y_hat = model(X, w)                     # output
#     loss = rss(y, y_hat)
#     loss.backward()                         # compute the gradient of the loss
#     w.data = w.data - step_size * w.grad    # do a gradient descent step
    
#     print('{},\t{:.2f},\t{}'.format(i, loss.item(), w.view(2).detach().numpy()))
    
#     # We need to zero the grad variable since the backward()
#     # call accumulates the gradients in .grad instead of overwriting.
#     # The detach_() is for efficiency. You do not need to worry too much about it.
#     w.grad.detach()
#     w.grad.zero_()

# print('\ntrue w\t\t', true_w.view(2).numpy())
# print('estimated w\t', w.view(2).detach().numpy())


# '''
# simple linear layer
# input 4x2 -> m (model) 2x3 (weights) -> output 4x3
# output = m(input)
# '''
# m = nn.Linear(2, 3)             # features x neurons
# input = torch.randn(4, 2)       # 4 samples x 2 features
# output = m(input)               # (4x2) x (2x3) = (4x3)
# print(input.size())             # torch.Size([4, 2])
# print(input)
# print(output.size())            # torch.Size([4, 3])
# print(output)

# '''
# Linear module
# y = x * (W.T) + b
# (2x4) = (2x3) * (3x4) + (1x4) --> (2x4) + (1x4). broadcast rule (1x4) becomes (2x4) then add
# '''
# d_in = 3
# d_out = 4
# linear_module = nn.Linear(d_in, d_out)              # 3 inputs x 4 neurons

# example_tensor = torch.tensor([[1.,2,3], [4,5,6]])  # inputs: 2 samples x 3 features
# # applys a linear transformation to the data
# transformed = linear_module(example_tensor)         # (2x3)x(3x4) = (2x4)
# print('example_tensor shape', example_tensor.shape)       # torch.Size([2, 3])
# print('transormed shape', transformed.shape)              # torch.Size([2, 4])
# print()
# print('We can see that the weights exist in the background\n')
# print('W:', linear_module.weight)                   # weights in (4x3) matrix, transpose of linear module shape
# print('b:', linear_module.bias)                     # bias in (1x4) matrix, 1 x neurons (4)
# print('transormed\n', transformed)
# print('transormed manual\n', example_tensor@linear_module.weight.T + linear_module.bias)    # values are the same


# '''
# activation function
# '''
# activation_fn = nn.ReLU()                       # ReLU module
# example_tensor = torch.tensor([-1.0, 1.0, 0.0]) # inputs
# activated = activation_fn(example_tensor)       # outputs after ReLU, values should be (0, 1, 0)
# print('example_tensor', example_tensor)         # tensor([-1.,  1.,  0.])
# print('activated', activated)                   # tensor([0., 1., 0.])

# '''
# multiple modules together
# 2 samples
# 3 features -> 4 neurons w/ tanh -> output layer w/ sigmoid
# (2x3) * (3x4) * (4x1) -> (2x1)
# '''
# d_in = 3
# d_hidden = 4
# d_out = 1
# model = torch.nn.Sequential(
#                             nn.Linear(d_in, d_hidden),  # 3 features x 4 neurons - 1st hidden layer
#                             nn.Tanh(),                  # tanh activation function
#                             nn.Linear(d_hidden, d_out), # 4 neurons x 1 output 
#                             nn.Sigmoid()                # sigmoid activation function
#                            )

# example_tensor = torch.tensor([[1.,2,3],[4,5,6]])       # 2 samples x 3 features
# transformed = model(example_tensor)
# print('transformed', transformed.shape)                 # transformed torch.Size([2, 1])
# print(transformed)

# '''
# loss function
# 1/3 * (target - input)**2 = 1/3 * ((1-0)**2 + (0-0)**2 + (-1-0)**2) = 0.667
# '''
# mse_loss_fn = nn.MSELoss()
# input = torch.tensor([[0., 0, 0]])
# target = torch.tensor([[1., 0, -1]])
# loss = mse_loss_fn(input, target)
# print(loss)

# '''
# torch.optim: gradient-based optimization methods
# available optimizers (the way to update parameters, e.g. w = w - eta*gradient)
#     - Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, RMSProp, Rprop, SGD
# model
#     x -> 1 neuron -> y_hat -- mse -- y
# optimizer
#     Stochastic gradient descent
# optim.zero_grad()
#     - gradients are accumulated and need to zero first
# '''
# # create a simple model
# model = nn.Linear(1, 1)

# # create a simple dataset
# X_simple = torch.tensor([[1.]])
# y_simple = torch.tensor([[2.]])

# # create our optimizer
# optim = torch.optim.SGD(model.parameters(), lr=1e-2)
# mse_loss_fn = nn.MSELoss()

# y_hat = model(X_simple)
# print('y_hat', y_hat)
# print('model params before:', model.weight) # actual weights
# loss = mse_loss_fn(y_hat, y_simple)
# optim.zero_grad()                   # zero gradients of the optimizer 
# loss.backward()                     # compute gradients
# optim.step()                        # do gradient descent with appointed optimizer 
# print('model params after:', model.weight)

# '''
# cross entropy loss function, which does softmax automatically at output
# input: (N, C), N: minibatch size, C: number of classes
# output: (N), true labels
# '''
# loss = nn.CrossEntropyLoss()

# inputs = torch.tensor([
#     [[-1., 1],[-1, 1],[1, -1]],     # raw scores correspond to the correct class
#     [[-3., 3],[-3, 3],[3, -3]],     # raw scores correspond to the correct class with higher confidence
#     [[1., -1],[1, -1],[-1, 1]],     # raw scores correspond to the incorrect class
#     [[3., -3],[3, -3],[-3, 3]]      # raw scores correspond to the incorrect class with incorrectly placed confidence
# ])

# # input = torch.tensor([[-1., 1],[-1, 1],[1, -1]]) # raw scores correspond to the correct class
# # input = torch.tensor([[-3., 3],[-3, 3],[3, -3]]) # raw scores correspond to the correct class with higher confidence
# # input = torch.tensor([[1., -1],[1, -1],[-1, 1]]) # raw scores correspond to the incorrect class
# # input = torch.tensor([[3., -3],[3, -3],[-3, 3]]) # raw scores correspond to the incorrect class with incorrectly placed confidence

# for input in inputs:
#     target = torch.tensor([1, 1, 0])
#     output = loss(input, target)        # tensor(0.1269), tensor(0.0025), tensor(2.1269), tensor(6.0025)
#     print(output)

'''
use CNN to blur an image
'''
image = np.array([0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.3803922 , 0.37647063, 0.3019608 ,0.46274513, 0.2392157 , 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.3529412 , 0.5411765 , 0.9215687 ,0.9215687 , 0.9215687 , 0.9215687 , 0.9215687 , 0.9215687 ,0.9843138 , 0.9843138 , 0.9725491 , 0.9960785 , 0.9607844 ,0.9215687 , 0.74509805, 0.08235294, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.54901963,0.9843138 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.7411765 , 0.09019608, 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.8862746 , 0.9960785 , 0.81568635,0.7803922 , 0.7803922 , 0.7803922 , 0.7803922 , 0.54509807,0.2392157 , 0.2392157 , 0.2392157 , 0.2392157 , 0.2392157 ,0.5019608 , 0.8705883 , 0.9960785 , 0.9960785 , 0.7411765 ,0.08235294, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.14901961, 0.32156864, 0.0509804 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.13333334,0.8352942 , 0.9960785 , 0.9960785 , 0.45098042, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.32941177, 0.9960785 ,0.9960785 , 0.9176471 , 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.32941177, 0.9960785 , 0.9960785 , 0.9176471 ,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.4156863 , 0.6156863 ,0.9960785 , 0.9960785 , 0.95294124, 0.20000002, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.09803922, 0.45882356, 0.8941177 , 0.8941177 ,0.8941177 , 0.9921569 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.94117653, 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.26666668, 0.4666667 , 0.86274517,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 ,0.9960785 , 0.9960785 , 0.9960785 , 0.9960785 , 0.5568628 ,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.14509805, 0.73333335,0.9921569 , 0.9960785 , 0.9960785 , 0.9960785 , 0.8745099 ,0.8078432 , 0.8078432 , 0.29411766, 0.26666668, 0.8431373 ,0.9960785 , 0.9960785 , 0.45882356, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.4431373 , 0.8588236 , 0.9960785 , 0.9490197 , 0.89019614,0.45098042, 0.34901962, 0.12156864, 0., 0.,0., 0., 0.7843138 , 0.9960785 , 0.9450981 ,0.16078432, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.6627451 , 0.9960785 ,0.6901961 , 0.24313727, 0., 0., 0.,0., 0., 0., 0., 0.18823531,0.9058824 , 0.9960785 , 0.9176471 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0.07058824, 0.48627454, 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.32941177, 0.9960785 , 0.9960785 ,0.6509804 , 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.54509807, 0.9960785 , 0.9333334 , 0.22352943, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.8235295 , 0.9803922 , 0.9960785 ,0.65882355, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0.9490197 , 0.9960785 , 0.93725497, 0.22352943, 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.34901962, 0.9843138 , 0.9450981 ,0.3372549 , 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.01960784,0.8078432 , 0.96470594, 0.6156863 , 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.01568628, 0.45882356, 0.27058825,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0.], dtype=np.float32)
image_torch = torch.from_numpy(image).view(1, 1, 28, 28)

gaussian_kernel = torch.tensor([[1., 2, 1],[2, 4, 2],[1, 2, 1]]) / 16.0

conv = nn.Conv2d(1, 1, 3)
# manually set the conv weight
conv.weight.data[:] = gaussian_kernel

convolved = conv(image_torch)

plt.title('original image')
plt.imshow(image_torch.view(28,28).detach().numpy())
plt.show()

plt.title('blurred image')
plt.imshow(convolved.view(26,26).detach().numpy())
plt.show()