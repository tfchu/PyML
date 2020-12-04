import numpy as np
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

features_t = np.concatenate((water_t, normal_t))    # n x i
water_label_t = np.ones((89, 1))
normal_label_t = np.zeros((65, 1))
labels_t = np.concatenate((water_label_t, normal_label_t), axis = 0)  # n x 1
features_v = np.concatenate((water_v, normal_v))    # n x i
water_label_v = np.ones((54, 1))
normal_label_v = np.zeros((52, 1))
labels_v = np.concatenate((water_label_v, normal_label_v), axis = 0)  # n x 1

features = np.concatenate((features_t, features_v), axis = 0)   # n x i
labels = np.concatenate((labels_t, labels_v), axis = 0) # n x 1

def main():
    #global features, labels
    #normalized_range = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #labels = labels.T[0]
    #features = normalized_range.fit_transform(features)
    #features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4)

    # print(features_train)
    # print(features_test)
    # print(labels_train)
    # print(labels_test)

    # fit(n x i, 1 x n)
    #clf = LogisticRegression().fit(features_train, labels_train)
    #clf_score = clf.score(features_test, labels_test)
    #print('Scikit score: {}'.format(clf_score))

    clf = LogisticRegression().fit(features_t, labels_t.T[0])
    predictions = clf.predict(features_v)
    diff = predictions - labels_v.T[0]
    print('Accuracy: {0:.0%}'.format(np.count_nonzero(diff==0)/len(diff)))
    print(diff)
    print(np.argwhere(diff != 0))

if __name__ == '__main__':
    main()