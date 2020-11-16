import numpy as np
from numpy.testing import assert_almost_equal
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''
https://leemeng.tw/essence-of-principal-component-analysis.html

PCA: principal component analysis
find 1 or more matrix that "most" represent N-D data X and reduce dimension based on the matrix, then get K-D feature L (lenient feature)
"most" representative means
- maximum variance of K-D L after dimension reduction
- minimum error (RE, reconstruction error) after restoring from K-D L to N-D X
note. in this case 2-D X, and 1-D L. i.e. N = 2, K = 1

20 samples [x1, x2, .. x20], each wtih 2 features [x11, x12], [x21, x22], ... 
[
    [ x11 x21 x31 ... ] <-- 1st feature
    [ x12 x22 x32 ... ] <-- 2nd feature
]

X = [[ 2.89  0.32  5.8  -6.52  3.94 -4.21  0.45  2.14  1.3  -4.98 -2.4  -3.1
   0.69 -1.59 -3.64 -0.24  6.81  4.63 -2.24 -0.06]
 [ 1.52  0.91  1.52 -0.88 -0.03 -1.26 -0.25  0.96 -0.89 -0.45 -0.88 -1.12
  -0.86  0.13 -1.53  0.51  2.66  1.28 -0.14 -1.19]]

@: 1D and >1D are different. @ can be replaced by .dot
|1|   |3|
|2| x |4| = 1x3+2x4 = 11
v = np.array([1, 2])
w = np.array([3,4])
v@w
11

|1 2|   |3 4|   |1x3+2x5 1x4+2x6|   |13 16|
|3 4| x |5 6| = |3x3+4x5 3x4+4x6| = |29 36|
v = np.array([[1, 2],[3,4]])
w = np.array([[3,4],[5,6]])
v@w
array([[13, 16],
       [29, 36]])
'''
def pca_demo():
    np.set_printoptions(precision=2)    # show 2 decimal places
    rng = np.random.RandomState(1)      # reproductivity
    
    '''
    init test data X: 20 samples, each with 2 features
    # X[0]: feature 1 of all samples, X[1]: feature 2 of all samples
    '''
    W = rng.rand(2, 2)              # random value of shape (2, 2) within [0, 1)

    # X_normal follow Gaussian distribution, 
    mean, sigma = 0, 5
    X_normal = rng.normal(loc=mean, scale=sigma, size=(2, 20))    # loc: mean, scale: stdev
    
    # plot X, looks Gussian if with enough samples
    # count, bins, ignored = plt.hist(X_normal.reshape((40,)), 30, density=True)
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mean)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    # plt.show()

    # randomize, normalize X
    X_orig = W @ X_normal  # W (2,2) X_normal (2,20) -> X_orig (2,20)
    X_mean = X_orig.mean(axis=1)[:, np.newaxis]
    X = X_orig - X_mean
    mean = X.mean(axis=1)

    # 測試 numerical 相等，確保樣本的平均已經為 0
    # 這類數值測試在你需要自己實作 ML 演算法時十分重要
    assert_almost_equal(0, mean)        # no assert

    print('X.shape:', X.shape, '\n')
    print(X)

    # plot X
    plt.style.use('seaborn')
    plt.scatter(X[0], X[1])
    plt.axis('equal')
    plt.grid()
    plt.show()

    '''
    Linear transform X to v (or P1), i.e. P1 @ X
    assume transformation matrix v is known
    note. 
        v (dot product or @) w = (length of projected w on v) x (length of v, or 1 for unit vector) = length of projected w
        P1 = [[0.9691344, 0.246533]] = principal base
        2D->1D transformation 
            standard base i_hat = [1 0] -> [0.97 0], j_hat = [0 1] -> [0.25 0], [0.97 0.25] is transformation matrix

    given random vector x [5.8 1.52], the projected length on P1 = P1 @ X
        = np.array([0.9691344, 0.246533]) @ np.array([5.8, 1.52])           # 5.99570968, shape ()
        or np.array([[0.9691344, 0.246533]])@(np.array([[5.8, 1.52]]).T)     # array([[5.99570968]]), shape (1, 1)
    '''
    v = np.array([0.9691344, 0.246533])         # a vector representing the transformation matrix
    print("v       :", v)  # shape: (2,)        # [0.97 0.25]
    assert_almost_equal(1, np.linalg.norm(v))   # Frobenius norm of a vector = 0.9691344**2 + 0.246533**2 = 1, a unit vector

    P1 = v[np.newaxis, :]  # shape: (1, 2)      # P1: project matrix (transformation matrix)
    print("P1      :", P1)                      # [[0.97 0.25]]
    L = P1 @ X                                  # (1,2) @ (2,20) = (1,20). L: latent (hidden) feature
    print("L[:, :4]:", L[:, :4])                # [[ 3.18  0.53  5.99 -6.53]]

    ''' use sklearn to get PCA
    '''
    # 最大化 reproductivity
    random_state = 9527

    # 使用 sklearn 實作的 PCA 將數據 X 線性地降到 1 維
    # 這邊值得注意的是 sklearn API 預期的輸入維度為
    # (n_samples, n_features), 輸出自然也是。
    pca_1d = PCA(1, random_state=random_state)
    L_sk = pca_1d.fit_transform(X.T).T          # X.T: (20, 2) 20 samples each with 2 features -pca()-> (20, 1) -T-> (1, 20)
    print('L_sk.shape:', L_sk.shape)            # (1, 20)
    print('L_sk:', L_sk[:, :4])

    # sklearn API 得到的結果跟我們手動計算結果相同
    assert_almost_equal(L_sk, L)                # no assert

    '''
    inverse transform and find RE
    '''
    # with API
    X_proj = pca_1d.inverse_transform(L.T).T
    reconstruction_error =  np.linalg.norm(X - X_proj, axis=0).sum()
    print(reconstruction_error)
    # manual calculate
    P1 = pca_1d.components_                     # [[0.97 0.25]]. 2 features positive correlated. f1 is 4 times more important than f2
    assert_almost_equal(P1.T @ L, X_proj)       # no assert, P1.T (2, 1) @ L (1, 20) -> (2, 20)
    print(X_proj[:, :4])

    '''
    covariance: same as np.cov(X) where X is (2, 20)
    f1: [f11 f21 f31 ...]
    f2: [f21 f22 f32 ...]
    '''
    def cov(f1, f2):
        m1 = np.sum(f1)/f1.size                 # same as np.mean(f1)
        m2 = np.sum(f2)/f2.size
        return (f1 - m1)@(f2 - m2) / (f1.size -1)
    var_f1 = cov(X[0], X[0])
    var_f2 = cov(X[1], X[1])
    cov_f1_f2 = cov(X[0], X[1])
    K = np.array([[var_f1, cov_f1_f2], [cov_f1_f2, var_f2]])        # covariance matrix, same as np.cov(X)
    print('covariance matrix:\n{}'.format(K))                       # [[13.35  3.27] [ 3.27  1.33]]

    '''
    eigenvector
    '''
    eig_vals, eig_vecs = np.linalg.eig(K)
    print(f"eig_vecs.shape:", eig_vecs.shape)   # (2, 2)
    print(eig_vecs)                             # [[ 0.97 -0.25]  [ 0.25  0.97]]
    print("eig_vals.shape:", eig_vals.shape)    # (2, )
    print(eig_vals)                             # [14.18  0.5 ]
    pc1 = eig_vecs[:, :1]                       # pricipal component 1, which is v (PC1) used at beginning of script
    print(pc1)

def main():
    pca_demo()

if __name__ == '__main__':
    main()