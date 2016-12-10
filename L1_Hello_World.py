'''
Created on Nov 23, 2016

@author: tfchu
'''

from sklearn import tree

if __name__ == '__main__':
    #pass
    # differentiate apple and oragne
    # apple: lighter, smooth
    # orange: heavier, bumpy

    #feature: 1st two columns,input to classifier
    #features = [[140, 'smooth'], [130, 'smooth'], [150, 'bumpy'], [170, 'bumpy']]
    #number: weight in gram
    #smooth: 1, bumpy: 0
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    #label: last column, output of classifier
    #labels = ['apple', 'apple', 'orange', 'orange']
    #apple: 0, orange: 1
    labels = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    # find patterns in data
    clf = clf.fit(features, labels)
    print(clf.predict([[150, 0], [200, 0], [130, 0]]))
    