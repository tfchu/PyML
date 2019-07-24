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

def main():
    print(arange())

if __name__ == '__main__':
    main()