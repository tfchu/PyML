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

def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        #pass
        
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            #label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions
        #pass
        
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
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
    #pros: relatively simple
    #cons: computational intensive, hard to represent relationships between features
    