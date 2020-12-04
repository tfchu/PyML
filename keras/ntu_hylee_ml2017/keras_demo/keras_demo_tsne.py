'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 500)               392500
_________________________________________________________________
dense_2 (Dense)              (None, 500)               250500
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5010
=================================================================
Total params: 648,010
Trainable params: 648,010
Non-trainable params: 0
'''

from keras.models import load_model
from keras.datasets import mnist
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from keras import backend as K

NUM_SAMPLES = 5000

def load_data():
    # mnist = pd.read_csv('../../datasets/mnist_train_mini.csv')
    # print(mnist)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    number = NUM_SAMPLES
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)

    feat_cols = [ 'pixel'+str(i) for i in range(x_train.shape[1]) ]   # pixel0, pixel1, ... pixel63
    df = pd.DataFrame(x_train,columns=feat_cols)
    df['y'] = y_train
    
    print(df.shape)             # (1000, 785)

    return df, feat_cols

def get_layer_output(layer_name='dense_1'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    number = NUM_SAMPLES
    x_train = x_train[0:number]
    y_train = y_train[0:number]    
    x_train = x_train.reshape(number, 28 * 28)

    model = load_model('models/keras_demo.h5')
    get_output = K.function([model.input], [model.get_layer(layer_name).output])
    layer_output = []
    for i in range(number):
        layer_output.append(get_output([x_train[i].reshape((1, 28 * 28))])[0])

    output = np.asarray(layer_output).reshape((NUM_SAMPLES, 500))          # original (1000, 1, 500) 
    print(output.shape)                     # (NUM_SAMPLES, 500)

    feat_cols = ['n'+str(i) for i in range(output.shape[1])]
    df = pd.DataFrame(output,columns=feat_cols)
    df['y'] = y_train
    
    print(df.shape)                         # (NUM_SAMPLES, 501)

    return df, feat_cols

def tsne(data):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    return tsne_results

def plot_sns(df, tsne_results):
    plt.figure(figsize=(16,10))

    df['tsne-2d-x'] = tsne_results[:,0]
    df['tsne-2d-y'] = tsne_results[:,1]

    sns.scatterplot(
        x="tsne-2d-x",          # column 'tsne-2d-x'
        y="tsne-2d-y",          # column 'tsne-2d-y
        hue="y",                # column 'y', Grouping variable that will produce points with different colors
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha='auto'     # 0.3
    )
    plt.show()

def plot_matplot(df, tsne_results):
    plt.figure(figsize=(16, 10))

    target_names = np.unique(df['y'])
    target_ids = range(len(target_names))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'

    # plot 10 times, each time plot a number
    for i, c, label in zip(target_ids, colors, target_names):        # e.g. i, c, label = (0, 'r', 0), (1, 'g', 1), ... 
        plt.scatter(tsne_results[df['y'] == i, 0], tsne_results[df['y'] == i, 1], c=c, label=label)     # plt.scatter(x, y, color, label)                                                                    
    plt.legend()
    plt.show()

def main():
    # df, feat_cols = load_data()
    df, feat_cols = get_layer_output('dense_2')

    data = df[feat_cols].values
    tsne_results = tsne(data)

    plot_sns(df, tsne_results)
    # plot_matplot(df, tsne_results)

if __name__ == '__main__':
    main()