"""
pick 10 values as training set
pick another 10 values as testing set

input (data set)
original cp, upgraded cp, w, h

model
1. y = b + w * xcp
2. y = b + w1 * xcp + w2 * xcp^2

output
find average error
plot upgraded cp vs original cp

training set
index: [2, 3, 17, 44, 63, 86, 149, 160, 163, 187]
old cp: [12.0, 13.0, 27.0, 54.0, 73.0, 96.0, 159.0, 170.0, 173.0, 197.0]
new cp: [48.0, 48.0, 63.0, 85.0, 102.0, 108.0, 113.0, 112.0, 110.0, 94.0]
height: [77.0, 69.0, 95.0, 16.0, 90.0, 39.0, 40.0, 97.0, 72.0, 20.0]
weight: [89.0, 67.0, 61.0, 99.0, 87.0, 27.0, 63.0, 62.0, 69.0, 84.0]

validation set
index: [5, 30, 49, 56, 62, 86, 131, 157, 159, 186]
old cp: [15.0, 40.0, 59.0, 66.0, 72.0, 96.0, 141.0, 167.0, 169.0, 196.0]
new cp: [50.0, 73.0, 90.0, 98.0, 94.0, 108.0, 116.0, 110.0, 108.0, 95.0]
height: [36.0, 53.0, 79.0, 93.0, 17.0, 10.0, 29.0, 62.0, 58.0, 25.0]
weight: [89.0, 52.0, 54.0, 95.0, 11.0, 58.0, 58.0, 24.0, 11.0, 72.0]
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

x_data = [12.0, 13.0, 27.0, 54.0, 73.0, 96.0, 159.0, 170.0, 173.0, 197.0]   # xcp
y_data = [48.0, 48.0, 63.0, 85.0, 102.0, 108.0, 113.0, 112.0, 110.0, 94.0]  # y_cap

validation_x_data = [15.0, 40.0, 59.0, 66.0, 72.0, 96.0, 141.0, 167.0, 169.0, 196.0]
validation_y_data = [50.0, 73.0, 90.0, 98.0, 94.0, 108.0, 116.0, 110.0, 108.0, 95.0]

# y = b + w * x
# y: estimated cp after evolution
# x: cp before evolution
# b, w: to be found
class First_Order_Model():
    def __init__(self):
        self.b = 0
        self.w = 0
    
    def model(self, x_datum, b, w):
        return b + w*x_datum

    # loss function of n-th pair of data
    def loss(self, y_datum, x_datum, b, w):
        return (y_datum - self.model(x_datum, b, w))**2

    # x-axis: b
    # y-axis: w
    # 
    #   |
    # w |
    #   |  * L(b=i, w=j) = Z[i][j]
    #   ----------------
    #           b
    # Z[i][j] where i is y-axis sweep (w), j is x-axis sweep (b)
    def plot_loss_function_contour(self):
        x = np.linspace(-100, 100, 200) # bias b
        y = np.linspace(-100, 100, 200) # weight w
        Z = np.zeros((len(y), len(x)))  # Loss function 
        X, Y = np.meshgrid(x, y)

        for i in range(len(y)):
            for j in range(len(x)):
                w = y[i]
                b = x[j]
                for k in range(len(x_data)):
                    #Z[i][j] = (y_data[k] - b - w * x_data[k])**2
                    Z[i][j] = self.loss(y_data[k], x_data[k], b, w)
        plt.contourf(X, Y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
        plt.xlabel(r'$b$')
        plt.ylabel(r'$w$')

    # with same learning rate lr, we cannot find best point
    # use differnt learning rate for b and w
    # then update learning rate for each cycle
    # new lr = lr / sqrt(b_lr or w_lr)
    def plot_gradient_descent_history(self):
        b = 0                       # initial b
        w = 0                       # initial w, chaneg b, w and best b,w is very close
        b_history = [b]
        w_history = [w]
        gradient_b_history = []
        gradient_w_history = []
        b_lr_history = []
        w_lr_history = []
        lr = 1
        b_lr = 0.0                  # separate learning rate (NEW)
        w_lr = 0.0
        iteration = 10          # 10K cannot find best point, use 100K (200K causes slow response, adjust b0, w0)
        for i in range(iteration):
            gradient_b = 0
            gradient_w = 0
            # calculate gradient
            for n in range(len(x_data)):
                gradient_b = gradient_b - 2*(y_data[n] - b - w*x_data[n])
                gradient_w = gradient_w - 2*(y_data[n] - b - w*x_data[n])*x_data[n]
            # track gradient
            gradient_b_history.append(gradient_b)
            gradient_w_history.append(gradient_w)
            # update learning rate (NEW)
            b_lr = b_lr + gradient_b**2
            w_lr = w_lr + gradient_w**2
            # update parameter: orignal is x - lr * gradient_x 
            b = b - lr/np.sqrt(b_lr) * gradient_b
            w = w - lr/np.sqrt(w_lr) * gradient_w
            # add lr history
            b_lr_history.append(lr/np.sqrt(b_lr))
            w_lr_history.append(lr/np.sqrt(w_lr))
            # add to parameter history
            b_history.append(b)
            w_history.append(w)
        self.b = b
        self.w = w
        print('best model: b {}, w {}'.format(self.b, self.w))
        #plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')

        # gradient descent trace
        # [theta1, theta2] = [a, b] - lr * [dL(a,b)/d(theta1), dL(a,b)/d(theta_2)]
        print('[new_b, new_w] = [b, w] - lr * [grad_b, grad_w]')
        print('-----------------------------------------------------------------')
        for i in range(iteration-1):
            print('b: [{}] = [{}] - {} * [{}]'.format(b_history[i+1], b_history[i], b_lr_history[i], gradient_b_history[i]))
            print('w: [{}] = [{}] - {} * [{}]'.format(w_history[i+1], w_history[i], w_lr_history[i], gradient_w_history[i]))
            print('-----------------------------------------------------------------')

    def plot_best_point(self):
        plt.plot([59.31], [0.30], 'x', ms=12, markeredgewidth=3, color='orange')

    def plot_error(self):
        x = validation_x_data
        y = list()              # how to do mass operation with list? 
        for i in range(len(validation_x_data)):
            y.append(self.b + self.w*validation_x_data[i])
        y_cap = validation_y_data
        # print data
        print('x, estimated y, actual y, error')
        # calculate average error e
        e = 0
        for i in range(len(validation_y_data)):
            print(validation_x_data[i], self.model(validation_x_data[i], self.b, self.w), validation_y_data[i], self.model(validation_x_data[i], self.b, self.w) - validation_y_data[i])
            e = e + self.model(validation_x_data[i], self.b, self.w) - validation_y_data[i]
        e = e / len(validation_x_data)
        print('average error: {}'.format(e))
        plt1, = plt.plot(x, y, color='orange')
        plt2, = plt.plot(x, y_cap, color='black')
        plt.grid()
        plt.legend((plt1, plt2), ('estimated', 'actual'))
        plt.xlabel('CP before evolution')
        plt.ylabel('CP after evolution')

    def plot(self):
        plt.show()

# y = b + w1 * x + w2 * x^2
class Second_Order_Model():
    def __init__(self):
        self.b = 0
        self.w1 = 0
        self.w2 = 0

    def model(self, x_datum, b, w1, w2):
        return b + w1*(x_datum) + w2*(x_datum**2)

    def loss(self, y_datum, x_datum, b, w1, w2):
        pass

    def gradient_descent(self):
        b = 0
        w1 = 0
        w2 = 0 
        lr = 10
        b_lr = 0.0                  # separate learning rate (NEW)
        w1_lr = 0.0
        w2_lr = 0.0
        iteration = 100000
        for i in range(iteration):
            grad_b = 0
            grad_w1 = 0
            grad_w2 = 0
            # calculate gradient
            for n in range(len(x_data)):
                grad_b = grad_b - 2*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2))
                grad_w1 = grad_w1 - 2*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2)) * x_data[n]
                grad_w2 =  grad_w2 - 2*(y_data[n] - b - w1*x_data[n] - w2*(x_data[n]**2)) * (x_data[n]**2)
            # update learning rate
            b_lr = b_lr + grad_b**2
            w1_lr = w1_lr + grad_w1**2
            w2_lr = w2_lr + grad_w2**2
            # update parameter
            b = b - lr/np.sqrt(b_lr) * grad_b
            w1 = w1 - lr/np.sqrt(w1_lr) * grad_w1
            w2 = w2 - lr/np.sqrt(w2_lr) * grad_w2
        print('b: {}, w1: {}, w2: {}'.format(b, w1, w2))
        self.b = b
        self.w1 = w1
        self.w2 = w2

    def plot_error(self):
        x = validation_x_data
        y = list()              # how to do mass operation with list? 
        for i in range(len(validation_x_data)):
            y.append(self.model(validation_x_data[i], self.b, self.w1, self.w2))
        y_cap = validation_y_data
        # print data
        print('x, estimated y, actual y, error')
        # calculate average error e
        e = 0
        for i in range(len(validation_y_data)):
            print(validation_x_data[i], self.model(validation_x_data[i], self.b, self.w1, self.w2), validation_y_data[i], self.model(validation_x_data[i], self.b, self.w1, self.w2) - validation_y_data[i])
            e = e + self.model(validation_x_data[i], self.b, self.w1, self.w2) - validation_y_data[i]
        e = e / len(validation_x_data)
        print('average error: {}'.format(e))
        plt1, = plt.plot(x, y, color='orange')
        plt2, = plt.plot(x, y_cap, color='black')
        plt.grid()
        plt.legend((plt1, plt2), ('estimated', 'actual'))
        plt.xlabel('CP before evolution')
        plt.ylabel('CP after evolution')
        plt.show()
    
def main():
    # 1st order model: y = b + wx
    # note. plot_error() has differnt x domain, cannot plot together with other plot functions
    fom = First_Order_Model()
    #fom.plot_loss_function_contour()
    fom.plot_gradient_descent_history()
    #fom.plot_best_point()
    #fom.plot_error()
    #fom.plot()

    # 2nd order model: y = b + w1*x + w1*x^2
    # som = Second_Order_Model()
    # som.gradient_descent()
    # som.plot_error()
    
if __name__ == '__main__':
    main()