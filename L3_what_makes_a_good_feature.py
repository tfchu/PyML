'''
Created on Nov 30, 2016

@author: tfchu
'''
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 500 dogs for each species
    greyhounds = 500
    labs = 500
    
    # feature: height
    grey_height = 28 + 4 * np.random.randn(greyhounds)
    lab_height = 24 + 4 * np.random.randn(labs)

    plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
    plt.show()
    
    #what makes good features
    # informative
    # independent
    # simple