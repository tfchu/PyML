from sklearn import datasets
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from time import time

'''
digits dataset
classes: 10 (0 ~ 9)
samples per class: ~ 180
samples total: 1797
dimensionality: 64 (8x8)
features: integer 0 ~ 16 (gray scale from 0 to 15/16?)

e.g. handwriting 0
digits.images[0] 
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
 
 digits.data[0]
 [ 0.  0.  5. 13.  9.  1.  0.  0.  0.  0. 13. 15. 10. 15.  5.  0.  0.  3.
 15.  2.  0. 11.  8.  0.  0.  4. 12.  0.  0.  8.  8.  0.  0.  5.  8.  0.
  0.  9.  8.  0.  0.  4. 11.  0.  1. 12.  7.  0.  0.  2. 14.  5. 10. 12.
  0.  0.  0.  0.  6. 13. 10.  0.  0.  0.]

digits.target[0]
0

digits.target_names
[0 1 2 3 4 5 6 7 8 9]
'''
# Load the digits data (8x8 image of a digit)
digits = datasets.load_digits()

print(digits.data.shape)            # (1797, 64)

# visualize the image
# plt.figure(1, figsize=(3, 3))
# plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()

# Take the first 500 data points: it's hard to see 1500 points
X = digits.data[:500]
y = digits.target[:500]

# Fit and transform with a TSNE
time_start = time()
tsne = TSNE(n_components=2, verbose=1, random_state=0)

# Project the data in 2D
# e.g. [[x1, y1], [x2, y2], ... [x500, y500]]
X_2d = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))

# Visualize the data
target_ids = range(len(digits.target_names))            # 0 ~ 9
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):        # e.g. i, c, label = (0, 'r', 0), (1, 'g', 1), ... 
    '''
    y == i: locations where y == i
    e.g. i = 0, location list where y = 0, i.e. [0, 10, 20, ...]
    X_2d[0], X_2d[10], X_2d[20], ... are plotted, and [0] is x value, [1] is y value
    for each sample, we have a TSNE point. Since we picked 500 samples, we have 500 TSNE points
    '''
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)     # plt.scatter(x, y, color, label)                                                                    
plt.legend()
plt.show()

# df_subset = {}
# df_subset['tsne-2d-one'] = X_2d[:,0]
# df_subset['tsne-2d-two'] = X_2d[:,1]

# sns.scatterplot(
#     x="tsne-2d-one", 
#     y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )