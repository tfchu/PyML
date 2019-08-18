'''
Created on Nov 23, 2016

@author: tfchu
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

#types of classifier
# artificial neural network
# super vector machine
# Lions
# Tigers
# Bears
# Oh my!
# decision tree

#https://en.wikipedia.org/wiki/Iris_flower_data_set

# 1. load data set
#target is a set of labels, such as setosa, versicolor, virginica in integer form 0, 1, 3
#data is a set of features, such as [ 5.1  3.5  1.4  0.2] in the order of ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
iris = load_iris()

print(iris.feature_names)   # iris metadata
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

for i in range(len(iris.target)):
    print('Example %3d: label %s, feature %s' % (i, iris.target[i], iris.data[i]))
    
# 2. train a Classifier
# testing data: removed from original data set, used to test the Classifier's accuracy
test_idx = [0, 50, 100]
#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)
#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 3. predict label for new flower
#test_data: a set of features used to predict test_target (a set of labels in integer)
#test the classifier
print(test_target)
print(test_data)
print(clf.predict(test_data))

# 4. visualize the tree
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, 
                     out_file=dot_data, 
                     feature_names=iris.feature_names, 
                     class_names=iris.target_names, 
                     filled=True, rounded=True, 
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')

print(test_data[0], test_target[0])
print(iris.feature_names, iris.target_names)
# conclusion: every question the tree asks must be about one of the features
# the better the features are, the better tree we can build