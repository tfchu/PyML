from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# load digits dataset
digits = load_digits()
X = digits.data / 16.0                                  # (1797 x 64) (i.e. 8x8 vector for each sample, 1797 samples)
y = digits.target                                       # (1797,). i.e. array of labels
print(X.shape, y.shape)                                 # (1797, 64) (1797,)

# populate pd dataframe
# row: each sample, column: normalized pixel value of each sample (0 ~ 63)
# added column 'y'
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]   # pixel0, pixel1, ... pixel63
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
print(df)

X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))     # (1797, 66). i.e. (1797, 64+2) due to added 'y' and 'label' columns

np.random.seed(42)                                      # Seed the generator
rndperm = np.random.permutation(df.shape[0])            # array([a b c ...]).randomly permute a sequence from 0 ~ 1796

# randomly preview some images
# plt.gray()
# fig = plt.figure( figsize=(16,7) )
# for i in range(0,15):
#     ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
#     ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((8,8)).astype(float))
# plt.show()

N = 1000
df_subset = df.loc[rndperm[:N],:].copy()                # randomly pick N samples from dataframe, assign to df_subset
data_subset = df_subset[feat_cols].values               # feat_col of eack sample to numpy array, array([[sample 1] [sample 2] [sample 3] ...], dtype=int64)
print(data_subset.shape)                                # (1000, 64)

# tsne
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# plot
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", 
    y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()