import numpy as np

# linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
# Return evenly spaced numbers over a specified interval. Use number of points
# e.g. 
# (0, 1, 2): return [0. 1.]
# (0, 1, 3): return [0. 0.5. 1.]
def linspace():
    return np.linspace(0, 1, 2)

# arange([start, ]stop, [step, ]dtype=None)
# Return evenly spaced values within a given interval. Use step size
# e.g. 
# (0, 1, 0.5): return [ 0.   0.5]
def arange():
    return np.arange(0, 1, 0.5)

# return an array of given shape filled with zeros
# zeros(shape, dtype=float, order='C')
# shape: int or tuple of ints. (x, y) creates x by y array
# e.g. zeros((3, 2))
# [[ 0.  0.]  -> [0][0], [0][1]
#  [ 0.  0.]  -> [1][0], [1][1]
#  [ 0.  0.]] -> [2][0], [2][1]
#
# Cartesian coordinate i.e. (x, y) vs zeros or z
# (x, y) is zero-filled with zeros(len(y), len(x))
# (x0, y0) = z[0][0], (x1, y0) = z[0][1], (x2, y0) = z[0][2]
# (x0, y1) = z[1][0], (x1, y1) = z[1][1], (x2, y1) = z[1][2]
# 
# y1 | [1][0] [1][1] [1][2]
# y0 | [0][0] [0][1] [0][2]
#    -----------------------
#        x0     x1     x2
def zeros():
    x = np.arange(0, 3, 1)  # 0, 1, 2
    y = np.arange(0, 2, 1)  # 0, 1, 2
    return np.zeros((len(y), len(x)))

# return coordinate matrice from coordinate vectors
# (
#   array([[0, 1, 2], [0, 1, 2]]), 
#   array([[0, 0, 0], [1, 1, 1]])
# )
# if sparse = True
# (
#   array([[0, 1, 2]]), 
#   array([[0], [1]])
# )
#
# 1 | (1, 0) (1, 1) (1, 2)
# 0 | (0, 0) (0, 1) (0, 2)
#    -----------------------
#        0      1      2
# 1st array contains all x values starting x=0/y=0: [0, 1, 2], [0, 1, 2]
# 2nd array contains all y values starting x=0/y=0: [0, 0, 0], [1, 1, 1]
def meshgrid():
    x = np.arange(0, 3, 1)  # 0, 1, 2
    y = np.arange(0, 2, 1)  # 0, 1
    X, Y = np.meshgrid(x, y, sparse=True)
    return X, Y

'''
x       y       *           dot         x.T
[[1 2]  [[5 6]  [[5 12]     [[19 22]    [[1 3]
 [3 4]]  [7 8]]  [21 32]]    [43 50]]    [2 4]]
u       v       *           dot         u.T
[9 10]  [11 12] [99 120]    219         [9 10]
w               w.T     dot(w, W.T)     dot(w.T, w)
[[1 2 3]]       [[1]    [[14]]          [[1 2 3]
                 [2]                     [2 4 6]
                 [3]]                    [3 6 9]]
note.
'*' is element-wise multiply, for inner product use np.dot(x, y) or x.dot(y)
T (transpose) of a 1-D array has no effect (u.T)
'''
def numpy_math():
    x = np.array([[1, 2], [3, 4]])  # 2D
    y = np.array([[5, 6], [7, 8]])

    u = np.array([9, 10])           # 1D
    v = np.array([11, 12])

    w = np.array([[1, 2, 3]])       # 2D
    
    print(x.shape)          # (2, 2), 2D array
    print(x * y)            # element-wise math
    print(np.dot(x, y))     # inner product
    print(u.shape)          # (2, ), 1D array
    print(u * v)
    print(np.dot(u, v))
    print(x.T)              # transpose
    print(u.T)
    print(w.shape)          # (1, 3), 2D array
    print(np.dot(w, w.T))
    print(np.dot(w.T, w))

'''
x       y               broadcasted x   broadcasted y   x + y
[[2]    [1 1 1 1 1 2]   [[2 2 2 2 2 2]  [[1 1 1 1 1 2]  [[ 3  3  3  3  3  4]
 [4]                     [4 4 4 4 4 4]   [1 1 1 1 1 2]   [ 5  5  5  5  5  6]
 [6]                     [6 6 6 6 6 6]   [1 1 1 1 1 2]   [ 7  7  7  7  7  8]
 [8]]                    [8 8 8 8 8 8]]  [1 1 1 1 1 2]]  [ 9  9  9  9  9 10]]

a       b               broadcasted a   broadcasted b   a * b
[1 2 3] [[10]           [[1 2 3]        [[10 10 10]     [[10 20 30]
         [20]            [1 2 3]         [20 20 20]      [20 40 60]
         [30]            [1 2 3]         [30 30 30]      [30 60 90]
         [40]]           [1 2 3]]        [40 40 40]]     [40 80 120]

'''
def array_broadcast():
    x = np.array([[2], [4], [6], [8]])
    y = np.array([1, 1, 1, 1, 1, 2])
    print(x + y)

    a = np.array([1, 2, 3])
    b = np.array([[10], [20], [30], [40]])
    #b = np.array([10, 20, 30, 40])             # # b[:, np.newaxis] if b = np.array([10, 20, 30 , 40])
    print(a * b)

def main():
    #numpy_math()
    array_broadcast()

if __name__ == '__main__':
    main()