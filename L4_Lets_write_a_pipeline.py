'''
Created on Dec 7, 2016

@author: tfchu
'''

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
    my_classifier = KNeighborsClassifier()
    
    my_classifier.fit(X_train, y_train)
    
    #4. make prediction on test data
    predictions = my_classifier.predict(X_test)
    #print(predictions)  #show a list of lables
    
    #5. check accuracy
    print(accuracy_score(y_test, predictions))
    
    #take away:
    #Learning: to use training data to adjust parameters of a model
    #Neural network:a more sophisticated type of classifier, like decision tree or simple line
    
    