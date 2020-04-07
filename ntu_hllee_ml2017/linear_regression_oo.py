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
from mpl_toolkits.mplot3d import Axes3D
import csv

# training data: feature 
# x_data (n x i) matrix, label n x 1 matrix
x_data = np.array([[12.0], [13.0], [27.0], [54.0], [73.0], [96.0], [159.0], [170.0], [173.0], [197.0]])   # xcp
#x_data = np.array([[12., 77., 89.], [13., 69., 67.], [27., 95., 61.], [54., 16., 99.], [73., 90., 87.], [96., 39., 27.], [159., 40., 63.], [170., 97., 62.], [173., 72., 69.], [197., 20., 84.]])
y_data = np.array([[48.0], [48.0], [63.0], [85.0], [102.0], [108.0], [113.0], [112.0], [110.0], [94.0]])  # y_cap

# validation data
x_data_v = np.array([[15.0], [40.0], [59.0], [66.0], [72.0], [96.0], [141.0], [167.0], [169.0], [196.0]])
#x_data_v = np.array([[15., 36., 89.], [40., 53., 52.], [59., 79., 54.], [66., 93., 95.], [72., 17., 11.], [96., 10., 58.], [141., 29., 58.], [167., 62., 24.], [169., 58., 11.], [196., 25., 72.]])
y_data_v = np.array([[50.0], [73.0], [90.0], [98.0], [94.0], [108.0], [116.0], [110.0], [108.0], [95.0]])

# i-features, n-samples
class First_Order_Model():
    def __init__(self):
        self.D = len(x_data[0])     # 1: 1 feature (+ 1 label)
        self.w = np.zeros((self.D, 1))
        self.b = 0.
        self.iterations = 30000
        self.lr = 1.
        self.w_lr = 1.
        self.b_lr = 1.
        self.cost_history = []
        self.w_history = []
        self.b_history = []

    # get model output for given b and w
    # model: y = b + w * x
    # w: i x 1, b: scalar
    # return 1 x 1 if x_data: 1 x i
    # return n x 1 if x_data: n x i
    def predict(self, x_data):
        # n x 1 = scalar + n x i dot i x 1
        return self.b + x_data.dot(self.w)

    # loss function of n-th pair of data
    # return scalar
    def cost_function(self):
        # np.sum(n x 1 - n x 1)
        return np.sum((y_data - self.predict(x_data))**2)

    # update weights (i x 1) and bias (scalar)
    def update_weights_bias(self):
        predictions = self.predict(x_data)
        # i x 1 = (i x n) dot (n x 1 - n x 1)
        grad_w = -2 * x_data.T.dot(y_data - predictions)
        # i x 1
        self.w_lr = self.w_lr + grad_w**2
        # i x 1
        self.w = self.w - self.lr/np.sqrt(self.w_lr) * grad_w

        # scalar = np.sum(1 x n - scalar - 1 x i dot i x n)
        grad_b = - 2*np.sum(y_data - predictions)
        self.b_lr = self.b_lr + grad_b**2
        self.b = self.b - self.lr/np.sqrt(self.b_lr) * grad_b
        
        if self.D == 1:
            self.w_history.append(np.asscalar(self.w))
            self.b_history.append(self.b)

    # with same learning rate lr, we cannot find best point
    # use differnt learning rate for b and w
    # then update learning rate for each cycle
    # new lr = lr / sqrt(b_lr or w_lr)
    def train(self):
        for i in range(self.iterations):
            self.update_weights_bias()

            cost = self.cost_function()
            self.cost_history.append(cost)
            # Log Progress
            if i % 1000 == 1:
                print("iter: " + str(i) + " cost: " + str(cost))        
        print('b*: {}, w*: {}'.format(self.b, self.w))
    
    def accuracy(self):
        # 1 x n
        predictions = self.predict(x_data_v)
        # np.sum [(1 x n - 1 x n) / (1 x n)] / n
        print('Accuracy: {0:.2%}'.format(np.sum((predictions - y_data_v) / y_data_v) / len(x_data_v)))
        print('Standard deviation: {:0.2f}'.format(np.sqrt(np.sum((predictions - y_data_v)**2) / len(x_data_v))))
        print('Actual Label: {}'.format(y_data_v.T.astype(int)))
        print(' Predictions: {}'.format(predictions.T.astype(int)))

    def plot(self):   
        plt.figure(1)
        plt.plot(np.arange(0, len(self.cost_history), 1), self.cost_history)
        plt.grid()
        plt.ylim([self.cost_history[-1] - 20, self.cost_history[-1] + 20])
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Cost vs Iteration')

        if self.D == 1:
            plt.figure(2)
            plt1, = plt.plot(x_data_v.T[0], y_data_v.T[0], color='black')
            plt2, = plt.plot(x_data_v.T[0], self.predict(x_data_v).T[0], color='orange')
            plt.grid()
            plt.legend((plt1, plt2), ('actual', 'estimated'))
            plt.xlabel('CP before evolution')
            plt.ylabel('CP after evolution')
            plt.title('Actual vs Estimate')

            fig = plt.figure(3)
            x = np.linspace(-100, 100, 200) # bias b
            y = np.linspace(-100, 100, 200) # weight w
            Z = np.zeros((len(y), len(x)))  # Loss function 
            X, Y = np.meshgrid(x, y)

            plt.plot([self.b], self.w[0], 'x', ms=12, markeredgewidth=3, color='orange')

            for i in range(len(y)):
                for j in range(len(x)):
                    self.w = np.array([[y[i]]])
                    self.b = x[j]
                    Z[i][j] = self.cost_function()
            #plt.contourf(X, Y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=plt.get_cmap('jet'))
            cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=plt.get_cmap('jet'))

            plt.plot(self.b_history, self.w_history, 'o-', ms=3, lw=1.5, color='black')
            plt.xlabel(r'$b$')
            plt.ylabel(r'$w$')

        plt.show()


class Second_Order_Model():
    def __init__(self):
        self.D = len(x_data[0])     # 1: 1 feature (+ 1 label)
        self.w1 = np.zeros((self.D, 1))
        self.w2 = np.zeros((self.D, 1))
        self.b = 0.
        self.iterations = 500000
        self.lr = 1.
        self.w1_lr = 1.
        self.w2_lr = 1.
        self.b_lr = 1.
        self.cost_history = []
        # self.w1_history = []
        # self.w2_history = []
        # self.b_history = []

    # model: y = b + w1 * x + w2 * x^2
    #   w1, w2: i x 1, b: scalar
    # return 1 x 1 if x_data: 1 x i
    # return n x 1 if x_data: n x i
    def predict(self, x_data):
        # n x 1 = scalar + n x i dot i x 1 + n x i dot i x 1
        return self.b + x_data.dot(self.w1) + ((x_data)**2).dot(self.w2)

    # return scalar
    def cost_function(self):
        # sum(n x 1 - n x 1)
        return np.sum((y_data - self.predict(x_data))**2)

    # update weights and bias
    def update_weights_bias(self):
        predictions = self.predict(x_data)

        # i x 1 = (n x i).T dot (n x 1 - n x 1)
        grad_w1 = -2 * x_data.T.dot(y_data - predictions)
        # i x 1
        self.w1_lr = self.w1_lr + grad_w1**2
        # i x 1
        self.w1 = self.w1 - self.lr/np.sqrt(self.w1_lr) * grad_w1

        # i x 1 = (n x i).T dot (n x 1 - n x 1)
        grad_w2 = -2 * (x_data**2).T.dot(y_data - predictions)
        # i x 1
        self.w2_lr = self.w2_lr + grad_w2**2
        # i x 1
        self.w2 = self.w2 - self.lr/np.sqrt(self.w2_lr) * grad_w2

        # scalar = np.sum(n x 1 - n x 1)
        grad_b = -2*np.sum(y_data - predictions)
        self.b_lr = self.b_lr + grad_b**2
        self.b = self.b - self.lr/np.sqrt(self.b_lr) * grad_b
        
        # if self.D == 1:
        #     self.w_history.append(np.asscalar(self.w))
        #     self.b_history.append(self.b)

    # with same learning rate lr, we cannot find best point
    # use differnt learning rate for b and w
    # then update learning rate for each cycle
    # new lr = lr / sqrt(b_lr or w_lr)
    def train(self):
        for i in range(self.iterations):
            self.update_weights_bias()

            cost = self.cost_function()
            self.cost_history.append(cost)
            # Log Progress
            if i % 10000 == 1:
                print("iter: " + str(i) + " cost: " + str(cost))        
        print('b*: {}, w1*: {}, w2*: {}'.format(self.b, self.w1, self.w2))

    def accuracy(self):
        # 1 x n
        predictions = self.predict(x_data_v)
        # np.sum [(1 x n - 1 x n) / (1 x n)] / n
        print('Accuracy: {0:.2%}'.format(np.sum((predictions - y_data_v) / y_data_v) / len(x_data_v)))
        print('Standard deviation: {:0.2f}'.format(np.sqrt(np.sum((predictions - y_data_v)**2) / len(x_data_v))))
        print('Actual Label: {}'.format(y_data_v.T.astype(int)))
        print(' Predictions: {}'.format(predictions.T.astype(int)))

    def plot(self):   
        plt.figure(1)
        plt.plot(np.arange(0, len(self.cost_history), 1), self.cost_history)
        plt.grid()
        plt.ylim([self.cost_history[-1] - 20, self.cost_history[-1] + 20])
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Cost vs Iteration')

        if self.D == 1:
            plt.figure(2)
            plt1, = plt.plot(x_data_v.T[0], y_data_v.T[0], color='black')
            plt2, = plt.plot(x_data_v.T[0], self.predict(x_data_v).T[0], color='orange')
            plt.grid()
            plt.legend((plt1, plt2), ('actual', 'estimated'))
            plt.xlabel('CP before evolution')
            plt.ylabel('CP after evolution')
            plt.title('Actual vs Estimate')

        plt.show()
    
def main():
    # 1st order model: y = b + wx
    fom = First_Order_Model()
    fom.train()
    fom.accuracy()
    fom.plot()

    # 2nd order model: y = b + w1*x + w1*x^2
    # som = Second_Order_Model()
    # som.train()
    # som.accuracy()
    # som.plot()
    
if __name__ == '__main__':
    main()