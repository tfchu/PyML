'''
Four files for logistic regression
    with_b  with_alr    file                result  accuracy    iter(min_loss)  w                   b
1   no      no          nob_noalr.py        bad
2   no      yes         nob.py              good    65.09%      ~20000          [[-0.0154, 0.0244]] 0.6509
3   yes     no          noalr.py            bad
4   yes     yes         test.py (this file) good    65.09%      ~50000          [[-0.0085, 0.0341]] -1.2162

*b: bias
*alr: adaptive learning rate
'''
import math
import numpy as np
import matplotlib.pyplot as plt
#from functions import sigmoid
from scipy.special import expit as sigmoid

# Training set: (Class 1) water 89 samples, (Class 2) normal 65 samples
water_t = np.array([[48, 50], [63, 65], [83, 85], [103, 135], [52, 65], [82, 95], [50, 40], [65, 50], [95, 70], [40, 50], [70, 80], [65, 40], [75, 100], [75, 130], [45, 45], [70, 70], [65, 45], [95, 85], [105, 25], [130, 50], [40, 70], [65, 95], [67, 35], [92, 65], [45, 70], [75, 100], [10, 15], [125, 60], [155, 70], [85, 85], [65, 110], [40, 90], [60, 115], [80, 55], [115, 65], [65, 44], [80, 59], [105, 79], [38, 56], [58, 76], [20, 20], [50, 60], [75, 90], [45, 25], [
                   85, 65], [75, 100], [95, 55], [55, 65], [65, 65], [105, 105], [40, 80], [95, 95], [75, 90], [70, 50], [85, 60], [110, 85], [150, 95], [30, 40], [50, 60], [70, 90], [30, 55], [50, 95], [30, 50], [90, 65], [120, 95], [140, 110], [70, 70], [90, 90], [48, 46], [78, 76], [80, 50], [120, 90], [15, 10], [60, 100], [70, 70], [40, 55], [60, 75], [80, 95], [64, 74], [104, 94], [84, 114], [90, 45], [30, 40], [100, 150], [150, 180], [51, 61], [66, 81], [86, 111], [85, 55]])
normal_t = np.array([[45, 35], [60, 50], [80, 70], [80, 135], [56, 25], [56, 25], [81, 50], [71, 40], [60, 31], [90, 61], [45, 45], [70, 85], [45, 40], [70, 65], [90, 58], [85, 35], [110, 60], [55, 60], [5, 35], [95, 40], [125, 60], [100, 40], [48, 48], [55, 45], [75, 65], [60, 85], [110, 65], [46, 35], [76, 45], [30, 36], [50, 86], [30, 40], [
                    70, 40], [80, 90], [70, 65], [80, 50], [130, 75], [80, 105], [95, 85], [20, 20], [80, 40], [10, 75], [30, 30], [70, 50], [55, 30], [85, 75], [60, 35], [80, 55], [160, 95], [51, 51], [71, 71], [91, 91], [20, 20], [45, 35], [65, 55], [60, 60], [40, 40], [115, 60], [70, 70], [90, 60], [55, 30], [75, 40], [120, 50], [45, 35], [85, 55]])
features_t = np.concatenate((water_t, normal_t))
water_label_t = np.ones((1, 89))
normal_label_t = np.zeros((1, 65))
labels_t = np.concatenate((water_label_t, normal_label_t), axis = 1)  # 1 x n: np.array([[1, 1, 1, ... 0]])

# Validation set: (Class 1) water 54 samples, (Class 2) normal 52 samples
water_v = np.array([[65, 60], [105, 85], [48, 57], [83, 92], [49, 49], [69, 69], [20, 60], [65, 105], [120, 150], [80, 80], [100, 100], [55, 63], [75, 83], [100, 108], [53, 53], [98, 98], [50, 50], [65, 65], [95, 85], [92, 80], [92, 80], [78, 53], [108, 83], [44, 44], [87, 87], [40, 65], [
                   60, 85], [75, 40], [72, 129], [72, 129], [56, 62], [63, 83], [95, 103], [145, 153], [52, 39], [105, 54], [60, 60], [53, 58], [73, 120], [110, 130], [54, 66], [69, 91], [74, 126], [20, 25], [140, 140], [53, 43], [63, 53], [40, 40], [70, 50], [35, 20], [125, 60], [60, 30], [105, 70], [75, 95]])
normal_v = np.array([[100, 60], [66, 44], [76, 54], [136, 54], [55, 42], [82, 64], [5, 15], [65, 92], [85, 40], [85, 80], [80, 135], [160, 80], [120, 120], [55, 35], [85, 60], [60, 25], [80, 35], [110, 45], [55, 36], [77, 50], [115, 65], [60, 60], [60, 80], [50, 40], [95, 65], [
                    60, 40], [100, 60], [110, 40], [83, 37], [123, 57], [77, 128], [128, 77], [36, 32], [56, 50], [50, 40], [50, 73], [68, 109], [80, 65], [38, 61], [55, 109], [75, 30], [85, 40], [120, 75], [70, 30], [110, 55], [75, 45], [125, 55], [60, 90], [95, 95], [95, 95], [115, 75], [60, 135]])
features_v = np.concatenate((water_v, normal_v))
water_label_v = np.ones((1, 54))
normal_label_v = np.zeros((1, 52))
labels_v = np.concatenate((water_label_v, normal_label_v), axis = 1)  # 1 x n: np.array([[1, 1, 1, ... 0]])

# # fake features: 50 samples each
# water_t = np.array([[161.0, 152.0], [180.0, 158.0], [157.0, 150.0], [120.0, 137.0], [140.0, 175.0], [143.0, 149.0], [140.0, 156.0], [120.0, 171.0], [126.0, 162.0], [158.0, 143.0], [161.0, 144.0], [175.0, 177.0], [169.0, 151.0], [120.0, 173.0], [172.0, 121.0], [150.0, 147.0], [125.0, 127.0], [156.0, 176.0], [148.0, 164.0], [149.0, 180.0], [120.0, 157.0], [174.0, 150.0], [163.0, 153.0], [176.0, 170.0], [178.0, 124.0], [179.0, 157.0], [138.0, 162.0], [179.0, 126.0], [127.0, 142.0], [161.0, 160.0], [145.0, 168.0], [163.0, 125.0], [155.0, 169.0], [161.0, 157.0], [169.0, 163.0], [136.0, 130.0], [173.0, 146.0], [133.0, 181.0], [135.0, 170.0], [138.0, 130.0], [148.0, 163.0], [162.0, 131.0], [142.0, 167.0], [134.0, 169.0], [140.0, 181.0], [130.0, 178.0], [135.0, 157.0], [181.0, 171.0], [132.0, 136.0], [168.0, 171.0]])
# normal_t = np.array([[61.0, 34.0], [35.0, 72.0], [20.0, 68.0], [28.0, 43.0], [42.0, 51.0], [63.0, 57.0], [44.0, 33.0], [54.0, 67.0], [46.0, 47.0], [60.0, 45.0], [47.0, 70.0], [51.0, 73.0], [46.0, 66.0], [45.0, 51.0], [30.0, 56.0], [27.0, 70.0], [72.0, 29.0], [79.0, 42.0], [26.0, 75.0], [25.0, 50.0], [31.0, 70.0], [31.0, 59.0], [56.0, 48.0], [22.0, 77.0], [45.0, 65.0], [21.0, 36.0], [80.0, 47.0], [23.0, 65.0], [24.0, 57.0], [31.0, 29.0], [76.0, 38.0], [69.0, 81.0], [58.0, 76.0], [47.0, 39.0], [26.0, 55.0], [38.0, 56.0], [66.0, 20.0], [56.0, 65.0], [40.0, 72.0], [59.0, 48.0], [65.0, 20.0], [55.0, 37.0], [34.0, 49.0], [26.0, 21.0], [75.0, 80.0], [23.0, 53.0], [25.0, 38.0], [25.0, 29.0], [42.0, 35.0], [54.0, 64.0]])
# features_t = np.concatenate((water_t, normal_t))
# water_label_t = np.ones((1, 50))
# normal_label_t = np.zeros((1, 50))
# labels_t = np.concatenate((water_label_t, normal_label_t), axis = 1)  # 1 x n: np.array([[1, 1, 1, ... 0]])

# water_v = np.array([[150, 150]])
# normal_v = np.array([[50, 50]])
# features_v = np.concatenate((water_v, normal_v))
# water_label_v = np.ones((1, 1))
# normal_label_v = np.zeros((1, 1))
# labels_v = np.concatenate((water_label_v, normal_label_v), axis = 1)  # 1 x n: np.array([[1, 1, 1, ... 0]])

# weights: 1 x i
# features: 1 x i or n x i
#   if 1 x i, return 1-element in the format np.array([[p]])
#   if n x i, return 1 x n, (w.x + b) for each sample
def predict(weights, features, bias):
    return sigmoid(weights.dot(features.T) + np.full((1, len(features)), bias))

# weights: 1 x i
# features: n x i
# lables: 1 x n
def update_weights_bias(weights, bias, lr, w_lr, b_lr):
    # 1 x n
    predictions_t = predict(weights, features_t, bias)
    # 1 x i = (1 x i) - (1 x n - 1 x n) dot (n x i)
    #w_grad = (predictions_t - labels_t).dot(features_t) / len(features_t)
    w_grad = (predictions_t - labels_t).dot(features_t)
    w_lr = w_lr + w_grad**2
    weights = weights - lr/np.sqrt(w_lr) * w_grad
    # scalar
    #b_grad = np.sum((predictions_t - labels_t)) / len(features_t)
    b_grad = np.sum((predictions_t - labels_t))
    b_lr = b_lr + b_grad**2
    bias = bias - lr/np.sqrt(b_lr) * b_grad
    return weights, bias, w_lr, b_lr

def cost_function(weights, bias):
    # 1 x n
    predictions_t = predict(weights, features_t, bias)
    # (1 x 1) = (1 x n) * (n x 1)
    class1_cost = -labels_t.dot(np.log(predictions_t.T))
    #print(class1_cost)
    # (1 x 1) = (1 x n) * (n x 1)
    class2_cost = (1-labels_t).dot(np.log(1-predictions_t.T))
    #print(np.log(1-predictions_t.T))
    # 1-element ([[p]])
    cost = (class1_cost + class2_cost) / len(labels_t)
    #cost = cost.sum() / len(labels_t)
    return cost

def train(weights, bias, lr, w_lr, b_lr, iters):
    cost_history = []

    for i in range(iters):
        weights, bias, w_lr, b_lr = update_weights_bias(weights, bias, lr, w_lr, b_lr)
        #print(weights)
        #print(weights, bias)

        # Calculate error for auditing purposes
        cost = cost_function(weights, bias)
        cost_history.append(cost)

        # Log Progress
        if i % 10000 == 0:
           print("iter: "+str(i) + " cost: "+str(cost))

    #return weights, cost_history
    return weights, bias

def accuracy(weights, bias):
    predictions = predict(weights, features_v, bias)
    #print(predictions)
    predicted_labels = (predictions >= 0.5).astype(int)
    # 1 x n: 0: correct prediction, 1: wrong prediction
    diff = predicted_labels - labels_v
    print(np.count_nonzero(diff==0)/len(diff[0]))

def plot_contour(weights, bias):
    x = np.linspace(0, 200, 50)        # attack
    y = np.linspace(0, 200, 50)        # sp_atk
    Z = np.zeros((len(y), len(x)))
    X, Y = np.meshgrid(x, y)

    for i in range(len(y)):
        for j in range(len(x)):
            #print('x: {}, y: {}'.format(j, i))
            Z[i][j] = predict(weights, np.array([[x[j], y[i]]]), bias)
    #print(Z[10])
    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
    CS = ax.contour(X, Y, Z, [0.5], colors='r')
    ax.clabel(CS, inline=1, fontsize=12, manual=[(175, 75)])

    ax.plot([water_v.T[0]], [water_v.T[1]], 'o', color='blue')
    ax.plot([normal_v.T[0]], [normal_v.T[1]], 'o', color='red')
    ax.set(xlabel = 'Attack', ylabel = 'Sp_Atk', title = 'Validation data set: water (blue), normal (red)')

    plt.show()

def main():
    w0 = np.array([[0, 0]])
    b0 = 0.
    lr = 1.
    w_lr = 1.
    b_lr = 1.
    iteration = 100000
    weights, bias = train(w0, b0, lr, w_lr, b_lr, iteration)
    print(weights, bias)

    accuracy(weights, bias)
    plot_contour(weights, bias)

# features: (n samples) x (m features array), e.g. 100 x 2
# weights: (n samples) x 1
# return probability array: array([[p]])
# def predict(features, weights):
#     '''
#     Returns 1D array of probabilities
#     that the class label == 1
#     '''
#     z = np.dot(features, weights)
#     return sigmoid(z)

# # n samples
# # return a cost value (scalar)
# def cost_function(features, labels, weights):
#     '''
#     Using Mean Absolute Error
#     Features:(100,3)
#     Labels: (100,1)
#     Weights:(3,1)
#     Returns 1D matrix of predictions
#     Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
#     '''
#     observations = len(labels)

#     predictions = predict(features, weights)

#     # Take the error when label=1
#     # n x 1 array
#     # print(predictions)
#     class1_cost = -labels*np.log(predictions)

#     # Take the error when label=0
#     # n x 1 array
#     class2_cost = (1-labels)*np.log(1-predictions)

#     # Take the sum of both costs
#     # a scalar
#     cost = class1_cost + class2_cost

#     # Take the average cost
#     cost = cost.sum() / observations

#     return cost


# def update_weights(features, labels, weights, lr):
#     '''
#     Vectorized Gradient Descent
#     Features:(200, 3)
#     Labels: (200, 1)
#     Weights:(3, 1)
#     '''
#     N = len(features)

#     # 1 - Get Predictions
#     predictions = predict(features, weights)

#     # 2 Transpose features from (200, 3) to (3, 200)
#     # So we can multiply w the (200,1)  cost matrix.
#     # Returns a (3,1) matrix holding 3 partial derivatives --
#     # one for each feature -- representing the aggregate
#     # slope of the cost function across all observations
#     gradient = np.dot(features.T,  predictions - labels)

#     # 3 Take the average cost derivative for each feature
#     gradient /= N

#     # 4 - Multiply the gradient by our learning rate
#     gradient *= lr

#     # 5 - Subtract from our weights to minimize cost
#     weights -= gradient

#     return weights

def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def classify(predictions):
    '''
    input  - N element array of predictions between 0 and 1
    output - N element array of 0s (False) and 1s (True)
    '''
    decision_boundary = np.vectorize(decision_boundary)
    return decision_boundary(predictions).flatten()



# def accuracy(predicted_labels, actual_labels):
#     diff = predicted_labels - actual_labels
#     return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right')
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()

if __name__ == '__main__':
    main()
