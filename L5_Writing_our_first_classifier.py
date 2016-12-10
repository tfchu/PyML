'''
Created on Dec 7, 2016

@author: tfchu
'''

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
from scipy.spatial import distance

#Euclidean formula c = sqrt(a^2 + b^2)
def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    #description
    # input the training data
    #param
    # X_train: a set of training data (fields), e.g. [X1, X2, X3, ...] where X is a list of fields [x1, x2, x3, x4 ...]
    # y_train: corresponding label for each training data (fields)
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        #pass
    
    #description
    # get the predictions for each test data
    #param
    # X_test: a list of test data
    #return
    # a list of predicted lables
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            #label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions
        #pass
    
    #description
    # Used to determine which label in training data (y_train) a test data (one of X_test) belongs to
    # by finding the shortest distance between the desired test data and training data (X_train), 
    # the label of the closest training data (one of X_train) is then the predicted result (label)
    # i.e. if the test data is closer to label A than to label B, then we say most likely the test data is label A (linear separation)
    #param
    # row: one test data, which is a list of fields
    #return
    # the predicted label of the test data
    def closest(self, row):
        #euc() finds the distance between: one set of test data, and one set of training data
        #start with 1st training data, assume it is the best (shortest) distance
        #then set the best index to index number of 1st training data, 0
        best_dist = euc(row, self.X_train[0])   
        best_index = 0
        #loop thru every training data set, and find the best (shortest) distance
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])    
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

if __name__ == '__main__':
    #pass
    #1. load iris data set
    iris = datasets.load_iris()
    
    #X: data (a set of fields like pedal height, width...)
    #y: target (a set of labels, like setosa, ...)
    #f(X) = y where f is the ideal function to convert the fields to a label
    #e.g. f([ 5.1  3.5  1.4  0.2]) = 'Setosa'
    #or in Python
    #def classify(features):
    #  logic
    #  return label
    X = iris.data
    y = iris.target
    
    #print(X)
    #print(y)
    
    #2. split iris data set into training data and test data
    #test data is to verify accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
    
    #3.1 train the classifier
    #type1: tree classifier
    #my_classifier = tree.DecisionTreeClassifier()
    #type2: KNeighborsClassifier
    #my_classifier = KNeighborsClassifier()
    #type3: ScrappyKNN(), this is implemented by ourselves
    my_classifier = ScrappyKNN()
    
    my_classifier.fit(X_train, y_train)
    
    #4. make prediction on test data
    predictions = my_classifier.predict(X_test)
    #print(predictions)  #show a list of lables
    
    #5. check accuracy
    print(accuracy_score(y_test, predictions))
    
    #take away:
    #K neighbor Classifier
    #accuracy > 90% but is random, depending on what the training and test data are
    #pros: relatively simple
    #cons: computational intensive, hard to represent relationships between features
    