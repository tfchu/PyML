import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 2-feature Training set: (Class 1) water 89 samples, (Class 2) normal 65 samples
water_t = np.array([[48, 50], [63, 65], [83, 85], [103, 135], [52, 65], [82, 95], [50, 40], [65, 50], [95, 70], [40, 50], [70, 80], [65, 40], [75, 100], [75, 130], [45, 45], [70, 70], [65, 45], [95, 85], [105, 25], [130, 50], [40, 70], [65, 95], [67, 35], [92, 65], [45, 70], [75, 100], [10, 15], [125, 60], [155, 70], [85, 85], [65, 110], [40, 90], [60, 115], [80, 55], [115, 65], [65, 44], [80, 59], [105, 79], [38, 56], [58, 76], [20, 20], [50, 60], [75, 90], [45, 25], [
                   85, 65], [75, 100], [95, 55], [55, 65], [65, 65], [105, 105], [40, 80], [95, 95], [75, 90], [70, 50], [85, 60], [110, 85], [150, 95], [30, 40], [50, 60], [70, 90], [30, 55], [50, 95], [30, 50], [90, 65], [120, 95], [140, 110], [70, 70], [90, 90], [48, 46], [78, 76], [80, 50], [120, 90], [15, 10], [60, 100], [70, 70], [40, 55], [60, 75], [80, 95], [64, 74], [104, 94], [84, 114], [90, 45], [30, 40], [100, 150], [150, 180], [51, 61], [66, 81], [86, 111], [85, 55]])
normal_t = np.array([[45, 35], [60, 50], [80, 70], [80, 135], [56, 25], [56, 25], [81, 50], [71, 40], [60, 31], [90, 61], [45, 45], [70, 85], [45, 40], [70, 65], [90, 58], [85, 35], [110, 60], [55, 60], [5, 35], [95, 40], [125, 60], [100, 40], [48, 48], [55, 45], [75, 65], [60, 85], [110, 65], [46, 35], [76, 45], [30, 36], [50, 86], [30, 40], [
                    70, 40], [80, 90], [70, 65], [80, 50], [130, 75], [80, 105], [95, 85], [20, 20], [80, 40], [10, 75], [30, 30], [70, 50], [55, 30], [85, 75], [60, 35], [80, 55], [160, 95], [51, 51], [71, 71], [91, 91], [20, 20], [45, 35], [65, 55], [60, 60], [40, 40], [115, 60], [70, 70], [90, 60], [55, 30], [75, 40], [120, 50], [45, 35], [85, 55]])

# Validation set: (Class 1) water 54 samples, (Class 2) normal 52 samples
water_v = np.array([[65, 60], [105, 85], [48, 57], [83, 92], [49, 49], [69, 69], [20, 60], [65, 105], [120, 150], [80, 80], [100, 100], [55, 63], [75, 83], [100, 108], [53, 53], [98, 98], [50, 50], [65, 65], [95, 85], [92, 80], [92, 80], [78, 53], [108, 83], [44, 44], [87, 87], [40, 65], [
                   60, 85], [75, 40], [72, 129], [72, 129], [56, 62], [63, 83], [95, 103], [145, 153], [52, 39], [105, 54], [60, 60], [53, 58], [73, 120], [110, 130], [54, 66], [69, 91], [74, 126], [20, 25], [140, 140], [53, 43], [63, 53], [40, 40], [70, 50], [35, 20], [125, 60], [60, 30], [105, 70], [75, 95]])
normal_v = np.array([[100, 60], [66, 44], [76, 54], [136, 54], [55, 42], [82, 64], [5, 15], [65, 92], [85, 40], [85, 80], [80, 135], [160, 80], [120, 120], [55, 35], [85, 60], [60, 25], [80, 35], [110, 45], [55, 36], [77, 50], [115, 65], [60, 60], [60, 80], [50, 40], [95, 65], [
                    60, 40], [100, 60], [110, 40], [83, 37], [123, 57], [77, 128], [128, 77], [36, 32], [56, 50], [50, 40], [50, 73], [68, 109], [80, 65], [38, 61], [55, 109], [75, 30], [85, 40], [120, 75], [70, 30], [110, 55], [75, 45], [125, 55], [60, 90], [95, 95], [95, 95], [115, 75], [60, 135]])

# # full-feature training set: [attack, sp_atk, defense, sp_def, speed]
# water_t = np.array([[48, 50, 65, 64, 43], [63, 65, 80, 80, 58], [83, 85, 100, 105, 78], [103, 135, 120, 115, 78], [52, 65, 48, 50, 55], [82, 95, 78, 80, 85], [50, 40, 40, 40, 90], [65, 50, 65, 50, 90], [95, 70, 95, 90, 70], [40, 50, 35, 100, 70], [70, 80, 65, 120, 100], [65, 40, 65, 40, 15], [75, 100, 110, 80, 30], [75, 130, 180, 80, 30], [45, 45, 55, 70, 45], [70, 70, 80, 95, 70], [65, 45, 100, 25, 40], [95, 85, 180, 45, 70], [105, 25, 90, 25, 50], [130, 50, 115, 50, 75], [40, 70, 70, 25, 60], [65, 95, 95, 45, 85], [67, 35, 60, 50, 63], [92, 65, 65, 80, 68], [45, 70, 55, 55, 85], 
# [75, 100, 85, 85, 115], [10, 15, 55, 20, 80], [125, 60, 79, 100, 81], [155, 70, 109, 130, 81], [85, 85, 80, 95, 60], [65, 110, 60, 95, 65], [40, 90, 100, 55, 35], [60, 115, 125, 70, 55], [80, 55, 90, 45, 55], [115, 65, 105, 70, 80], [65, 44, 64, 48, 43], [80, 59, 80, 63, 58], [105, 79, 100, 83, 78], [38, 56, 38, 56, 67], [58, 76, 58, 76, 67], [20, 20, 50, 50, 40], [50, 60, 80, 80, 50], [75, 90, 75, 100, 70], [45, 25, 45, 25, 15], [85, 65, 85, 65, 35], [75, 100, 80, 110, 30], [95, 55, 85, 55, 85], [55, 65, 95, 95, 35], [65, 65, 35, 35, 65], [105, 105, 75, 75, 45], [40, 80, 70, 140, 70], 
# [95, 95, 95, 95, 85], [75, 90, 115, 115, 85], [70, 50, 50, 50, 40], [85, 60, 70, 70, 50], [110, 85, 90, 90, 60], [150, 95, 110, 110, 70], [30, 40, 30, 50, 30], [50, 60, 50, 70, 50], [70, 90, 70, 100, 70], [30, 55, 30, 30, 85], [50, 95, 100, 70, 65], [30, 50, 32, 52, 65], [90, 65, 20, 20, 65], [120, 95, 40, 40, 95], [140, 110, 70, 65, 105], [70, 70, 35, 35, 60], [90, 90, 45, 45, 60], [48, 46, 43, 41, 60], [78, 76, 73, 71, 60], [80, 50, 65, 35, 35], [120, 90, 85, 55, 55], [15, 10, 20, 55, 80], [60, 100, 79, 125, 81], [70, 70, 70, 70, 70], [40, 55, 50, 50, 25], [60, 75, 70, 70, 45], 
# [80, 95, 90, 90, 65], [64, 74, 85, 55, 32], [104, 94, 105, 75, 52], [84, 114, 105, 75, 52], [90, 45, 130, 65, 55], [30, 40, 55, 65, 97], [100, 150, 90, 140, 90], [150, 180, 90, 160, 90], [51, 61, 53, 56, 40], [66, 81, 68, 76, 50], [86, 111, 88, 101, 60], [85, 55, 60, 60, 71]])

# normal_t = np.array([[45, 35, 40, 35, 56], [60, 50, 55, 50, 71], [80, 70, 75, 70, 101], [80, 135, 80, 80, 121], [56, 25, 35, 35, 72], [56, 25, 35, 35, 72], [81, 50, 60, 70, 97], [71, 40, 70, 80, 77], [60, 31, 30, 31, 70], [90, 61, 65, 61, 100], [45, 45, 20, 25, 20], [70, 85, 45, 50, 45], [45, 40, 35, 40, 90], [70, 65, 60, 65, 115], [90, 58, 55, 62, 60], [85, 35, 45, 35, 75], [110, 60, 70, 60, 110], [55, 60, 75, 75, 30], [5, 35, 5, 105, 50], [95, 40, 80, 80, 90], [125, 60, 100, 100, 100], [100, 40, 95, 70, 110], [48, 48, 48, 48, 48], [55, 45, 50, 65, 55], [75, 65, 70, 85, 75], 
# [60, 85, 70, 75, 40], [110, 65, 65, 110, 30], [46, 35, 34, 45, 20], [76, 45, 64, 55, 90], [30, 36, 30, 56, 50], [50, 86, 50, 96, 70], [30, 40, 15, 20, 15], [70, 40, 55, 55, 85], [80, 90, 65, 65, 85], [70, 65, 70, 65, 45], [80, 50, 50, 50, 40], [130, 75, 75, 75, 55], [80, 105, 90, 95, 60], [95, 85, 62, 65, 85], [20, 20, 35, 45, 75], [80, 40, 105, 70, 100], [10, 75, 10, 135, 55], [30, 30, 41, 41, 60], [70, 50, 61, 61, 100], [55, 30, 30, 30, 85], [85, 75, 60, 50, 125], [60, 35, 60, 35, 30], [80, 55, 80, 55, 90], [160, 95, 100, 65, 100], [51, 51, 23, 23, 28], [71, 71, 43, 43, 48], 
# [91, 91, 63, 73, 68], [20, 20, 40, 40, 20], [45, 35, 45, 35, 50], [65, 55, 65, 55, 90], [60, 60, 60, 60, 60], [40, 40, 60, 75, 50], [115, 60, 60, 60, 90], [70, 70, 70, 70, 70], [90, 60, 70, 120, 40], [55, 30, 30, 30, 60], [75, 40, 50, 40, 80], [120, 50, 70, 60, 100], [45, 35, 40, 40, 31], [85, 55, 60, 60, 71]])

# water_v = np.array([[65, 60, 35, 30, 85], [105, 85, 55, 50, 115], [48, 57, 48, 62, 34], [83, 92, 68, 82, 39], [49, 49, 56, 61, 66], [69, 69, 76, 86, 91], [20, 60, 50, 120, 50], [65, 105, 107, 107, 86], [120, 150, 100, 120, 100], [80, 80, 80, 80, 80], [100, 100, 100, 100, 100], [55, 63, 45, 45, 45], [75, 83, 60, 60, 60], [100, 108, 85, 70, 70], [53, 53, 48, 48, 64], [98, 98, 63, 63, 101], [50, 50, 40, 40, 64], [65, 65, 55, 55, 69], [95, 85, 75, 75, 74], [92, 80, 65, 55, 98], [92, 80, 65, 55, 98], [78, 53, 103, 45, 22], [108, 83, 133, 65, 32], [44, 44, 50, 50, 55], [87, 87, 63, 63, 98], [40, 65, 50, 85, 40], [60, 85, 70, 105, 60], [75, 40, 80, 45, 65], [72, 129, 90, 90, 108], [72, 129, 90, 90, 108], [56, 62, 40, 44, 71], [63, 83, 52, 56, 97], [95, 103, 67, 71, 122], [145, 153, 67, 71, 132], [52, 39, 67, 56, 50], [105, 54, 115, 86, 68], [60, 60, 60, 60, 30], [53, 58, 62, 63, 44], [73, 120, 88, 89, 59], [110, 130, 120, 90, 70], [54, 66, 54, 56, 40], [69, 91, 69, 81, 50], [74, 126, 74, 116, 60], [20, 25, 20, 25, 40], [140, 140, 130, 135, 30], [53, 43, 62, 52, 45], [63, 53, 152, 142, 35], [40, 40, 52, 72, 27], [70, 50, 92, 132, 42], [35, 20, 40, 30, 80], [125, 60, 140, 90, 40], [60, 30, 130, 130, 5], [105, 70, 70, 70, 92], [75, 95, 115, 130, 85]])
# normal_v = np.array([[100, 60, 66, 66, 115], [66, 44, 44, 56, 85], [76, 54, 84, 96, 105], [136, 54, 94, 96, 135], [55, 42, 42, 37, 85], [82, 64, 64, 59, 112], [5, 15, 5, 65, 30], [65, 92, 45, 42, 91], [85, 40, 40, 85, 5], [85, 80, 95, 95, 50], [80, 135, 70, 75, 90], [160, 80, 110, 110, 100], [120, 120, 120, 120, 120], [55, 35, 39, 39, 42], [85, 60, 69, 69, 77], [60, 25, 45, 45, 55], [80, 35, 65, 65, 60], [110, 45, 90, 90, 80], [55, 36, 50, 30, 43], [77, 50, 62, 42, 65], [115, 65, 80, 55, 93], [60, 60, 86, 86, 50], [60, 80, 126, 126, 50], [50, 40, 40, 40, 75], [95, 65, 60, 60, 115], [60, 40, 50, 50, 75], [100, 60, 70, 70, 95], [110, 40, 95, 95, 55], [83, 37, 50, 50, 60], [123, 57, 75, 75, 80], [77, 128, 77, 128, 90], [128, 77, 90, 77, 128], [36, 32, 38, 36, 57], [56, 50, 77, 77, 78], [50, 40, 43, 38, 62], [50, 73, 58, 54, 72], [68, 109, 72, 66, 106], [80, 65, 60, 90, 102], [38, 61, 33, 43, 70], [55, 109, 52, 94, 109], [75, 30, 30, 30, 65], [85, 40, 50, 50, 75], [120, 75, 75, 75, 60], [70, 30, 30, 30, 45], [110, 55, 60, 60, 45], [75, 45, 50, 50, 50], [125, 55, 80, 60, 60], [60, 90, 80, 110, 60], [95, 95, 95, 95, 59], [95, 95, 95, 95, 95], [115, 75, 65, 95, 65], [60, 135, 85, 91, 36]])

features_t = np.concatenate((water_t, normal_t))
water_label_t = np.ones((89, 1))
normal_label_t = np.zeros((65, 1))
labels_t = np.concatenate((water_label_t, normal_label_t), axis = 0)  # n x 1
features_v = np.concatenate((water_v, normal_v))
water_label_v = np.ones((54, 1))
normal_label_v = np.zeros((52, 1))
labels_v = np.concatenate((water_label_v, normal_label_v), axis = 0)  # n x 1

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

# i features for each sample
# n samples
class Logistic_Regression():
    def __init__(self):
        self.D = len(features_t[0])     # number of features
        #self.weights = np.array([[0, 0, 0, 0, 0]])
        self.weights = np.zeros((self.D, 1))   # 1 x i, i.e. a weight for each feature
        # scalar for bias, lr, iteration
        self.bias = 0.
        self.lr = 1.
        self.w_lr = 1.
        self.b_lr = 1.
        self.iteration = 50000
        self.cost_history = []

    # model (sigmoid function)
    # predict whether given sample(s) (described with features) is class 1 (>= 0.5) or class 2 (< 0.5)
    # note. weights: i x 1
    # features: 1 x i for 1 sample, or n x i for all samples
    # return sigmoid(w.x + b) for each sample
    #   if features is 1 x i, return 1 x 1, e.g. [[0.3]]
    #   if features is n x i, return n x 1, e.g. [[0.3], [0.2], [0.5], ...]]
    def predict(self, features):
        # n x i dot i x 1 + scalar
        #return sigmoid(features.dot(self.weights) + np.full((len(features), 1), self.bias))
        return sigmoid(features.dot(self.weights) + self.bias)

    # calculate cost function value for each training iteration
    # return a scalar
    def cost_function(self):
        # n x 1 = predict(n x i)
        predictions_t = self.predict(features_t)
        # (1 x 1) = (n x 1).T dot (n x 1)
        class1_cost = labels_t.T.dot(np.log(predictions_t))
        # (1 x 1) = (n x 1).T dot (n x 1)
        class2_cost = (1-labels_t).T.dot(np.log(1-predictions_t))
        # (1 x 1) matrix
        cost = -(class1_cost + class2_cost)
        #cost = -(class1_cost + class2_cost) / len(labels_t)
        return np.asscalar(cost)

    # update weights and bias for each training iteration
    # note. weights: i x 1, features: n x i, labels: n x 1
    def update_weights_bias(self):
        # n x 1
        predictions_t = self.predict(features_t)
        # i x 1 = (n x i).T dot (n x 1 - n x 1) 
        w_grad = features_t.T.dot(predictions_t - labels_t)
        #w_grad = features_t.T.dot(predictions_t - labels_t) / len(features_t)
        self.w_lr = self.w_lr + w_grad**2
        self.weights = self.weights - self.lr/np.sqrt(self.w_lr) * w_grad
        # 1 x 1: bias is acutally a scalar
        b_grad = np.sum((predictions_t - labels_t))
        #b_grad = np.sum((predictions_t - labels_t)) / len(features_t)
        self.b_lr = self.b_lr + b_grad**2
        self.bias = self.bias - self.lr/np.sqrt(self.b_lr) * b_grad

    # find the local minimum, i.e. best weights (i x 1) and bias (scalar)
    def train(self):
        for i in range(self.iteration):
            self.update_weights_bias()
            # Record cost value
            cost = self.cost_function()
            self.cost_history.append(cost)
            # Log Progress
            if i % 10000 == 1:
                print("iter: "+str(i) + " cost: "+str(cost))
        print('weights: {}, bias: {}'.format(self.weights, self.bias))

    # find accuracy given validation data set
    # print accuracy, e.g. 65%, and error list
    def accuracy(self, print_errors = False):
        # n x 1
        predictions = self.predict(features_v)
        # n x 1: class 1 is 1, class 2 is 0
        # any value >= 0.5 is class 1, < 0.5 is class 2
        predicted_labels = (predictions >= 0.5).astype(int)
        # n x 1: 0: correct prediction (1-1, or 0-0), 1: wrong prediction (1-0 or 0-1)
        diff = predicted_labels - labels_v
        # 1 x n
        self.mydiff = diff.T
        print('My Accuracy: {0:.0%}'.format(np.count_nonzero(diff==0)/len(diff)))
        if print_errors:
            print('Error list:')
            error_indexes = np.argwhere(diff != 0).T[0]
            print(error_indexes)
            for index in error_indexes:
                print('{}: label {}, prediction {}'.format(features_v[index], int(labels_v[index][0]), predicted_labels[index][0]))

    def sklearn_accuracy(self):
        # fit(n x i, 1 x n)
        clf = LogisticRegression(solver='lbfgs').fit(features_t, labels_t.T[0])
        # 1 x n = predict(n x i)
        predictions = clf.predict(features_v)
        # 1 x n
        diff = predictions - labels_v.T[0]
        self.skdiff = diff
        print('Sk Accuracy: {0:.0%}'.format(np.count_nonzero(diff==0)/len(diff)))
    
    def compare(self):
        print('Diff list:')
        # 1 x n
        diff = self.mydiff - self.skdiff
        for i in np.argwhere(diff != 0).T[1]:
            print('{}: label {}, prediction (my - sk) {}'.format(features_v[i], int(labels_v[i][0]), diff[0][i]))

    # plot prediction contour + validation data set only for 2 features (i = 2)
    # plot cost vs iteration regardless of feature count 
    def plot(self):
        plt_num = 1
        # cost plot
        plt.figure(plt_num)
        plt_num = plt_num + 1
        plt.plot(np.arange(0, len(self.cost_history), 1), self.cost_history)
        plt.grid()
        plt.ylim([self.cost_history[-1] - 20, self.cost_history[-1] + 20])
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.title('Cost vs Iteration')  
        
        # contour plot
        if self.D == 2:
            x = np.linspace(0, 200, 50)        # attack
            y = np.linspace(0, 200, 50)        # sp_atk
            Z = np.zeros((len(y), len(x)))
            X, Y = np.meshgrid(x, y)

            for i in range(len(y)):
                for j in range(len(x)):
                    #print('x: {}, y: {}'.format(j, i))
                    Z[i][j] = self.predict(np.array([[x[j], y[i]]]))
            #print(Z[10])

            fig, ax = plt.subplots()
            fig = plt.figure(plt_num)
            plt_num = plt_num + 1
            ax.contourf(X, Y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
            CS = ax.contour(X, Y, Z, [0.5], colors='r')
            ax.clabel(CS, inline=1, fontsize=12, manual=[(175, 75)])

            ax.plot([water_v.T[0]], [water_v.T[1]], 'o', color='blue')
            ax.plot([normal_v.T[0]], [normal_v.T[1]], 'o', color='red')
            ax.set(xlabel = 'Attack', ylabel = 'Sp_Atk', title = 'Validation data set: water (blue), normal (red)')
        
        plt.show()

def main():
    lr = Logistic_Regression()
    lr.train()
    lr.accuracy()
    lr.sklearn_accuracy()
    lr.compare()
    # lr.plot()

if __name__ == '__main__':
    main()
