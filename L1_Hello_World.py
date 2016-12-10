'''
Created on Nov 23, 2016

How to use Decision tree classifier to distinguish apple and orange

features to differentiate apple and orange
(each column is a feature, last column is a label, each row is an example)
Weight    Texture    Label
lighter   smooth     Apple
heavier   bumpy      Orange

or in general
x1    x2    y    --> f(X) = f(x1, x2) = y, goal is to find a function g() which is close to f()

or in Python
def classify(features):
    logic
    return label

example
Weight    Texture    Label
140       Smooth(1)  Apple(0)
130       Smooth(1)  Apple(0)
150       Bumpy(0)   Orange(1)
170       Bumpy(0)   Orange(1)

supervised learning: create a classifier by finding patterns in examples (begin with examples of the problem to solve)

@author: tfchu
'''

from sklearn import tree

if __name__ == '__main__':
    #pass

    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)
    print(clf.predict([[150, 0], [200, 0], [130, 0]]))
    