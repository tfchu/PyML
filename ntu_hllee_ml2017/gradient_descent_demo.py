import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
#%matplotlib inline
import random as random
import numpy as np
import csv
	
x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

# sweep an arbitrary range of b and w to plot loss function L(f)
# L(f) = sum(y_data[n] - (b + w * x_data[n]))^2 where n = 0 ~ 9, and b = x, w = y
# each pair of (b, w) give one point of L(f)
x = np.arange(-200,-100,1)      # bias b: 100 elements
#x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)         # weight w: 100 elements
Z = np.zeros((len(y), len(x)))  # loss function L. np.zeros(shape) where shape is int or tuples of ints. 2D array here
                                # len(y): number of arrays. len(x): number of elements in an array
X, Y = np.meshgrid(x, y)        # [[-200, -199, ... -101] [-200, -199, ... -101] ... [-5, -5, ...] [-4.9, -4.9, ...]]

# Z = [
#       [Z[0][0] Z[0][1] Z[0][2] ...]   length = len(x)
#       [Z[1][0] Z[1][1] Z[1][2] ...]   length = len(x)
#       ...
# ]
# where 
# Z[0][0] = L(b=x[0], w=y[0]), Z[0][1] = L(b=x[1], w=y[0]), Z[0][2] = L(b=x[2], w=y[0]), ...
# Z[1][0] = L(b=x[0], w=y[1]), Z[1][1] = L(b=x[1], w=y[1]), Z[1][2] = L(b=x[2], w=y[1]), ...
# or Z[m][n] = L(b=x[n], w=y[m])
for i in range(len(x)):         # b
    for j in range(len(y)):     # w
        b = x[i]
        w = y[j]
        #Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n])**2
            Z[j][i] = Z[j][i]/len(x_data)
# end of b and w sweep

# apply grandient descent to find the best (b, w)
# ydata = b + w * xdata 
b = -120    # initial b
w = -4      # initial w
lr = 1      # learning rate, try 0.000001, 0.00001, ... 
iteration = 100000

# use different learning rate for b and w (added as if use same lr, we cannot find optimal point)
b_lr = 0.0
w_lr = 0.0

# Store initial values for plotting.
b_history = [b]
w_history = [w]

# Iterations
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):        
        b_grad = b_grad  - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad  - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
    b_lr = b_lr + b_grad**2
    w_lr = w_lr + w_grad**2
    
    # Update parameters.
    b = b - lr/np.sqrt(b_lr) * b_grad 
    w = w - lr/np.sqrt(w_lr) * w_grad
    
    # Store parameters for plotting
    b_history.append(b)
    w_history.append(w)

print(b, w)

# plot the figure
# plot b, w sweep
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
# plot best (b, w) point, i.e. b, w at iteration 10K (last point)
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
# plot steps of gradient descent
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()