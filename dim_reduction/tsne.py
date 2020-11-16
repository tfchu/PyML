# import swat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# # Create CAS Connection
# conn = swat.CAS(host, portnum, protocol='http')
# conn.sessionProp.setSessOpt(messageLevel='NONE') # Suppress CAS Messages

# # Load CAS Action Sets
# conn.loadactionset('pca')
# conn.loadactionset('tsne')

# Read in Data
digits = pd.read_csv('../datasets/digits.csv')
mnist = pd.read_csv('../datasets/mnist_train_mini.csv')

display(digits.shape)
display(mnist.shape)