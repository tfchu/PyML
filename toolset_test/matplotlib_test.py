'''
* tutorial
(TBD) https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67
(TBD) https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
(TBD) https://towardsdatascience.com/what-is-a-positive-definite-matrix-181e24085abd
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import cm

# y = f(x) = x
def plot_x():
    n = 8
    #y = np.zeros(n)
    x = np.linspace(0, 10, n)
    y = x
    plt.plot(x, y, 'o')     # dotted line with 'o', straight solid line without 'o'
    plt.ylim([-0.5, 10])
    plt.grid()
    plt.show()

def test_plot():
    plt.plot([0.73, 0.05], [0.05, 0.73], 'o', color = 'red')
    plt.plot([0.27], [0.27], 'o', color = 'blue')
    plt.grid()
    plt.show()

# y = ax^2 + bx + c
def plot_ax2_bx_c():
    n = 100
    x = np.linspace(-10, 10, n)
    a = 1
    b = 2
    c = 3
    y = a*(x**2) + b*x + c
    dydx = np.abs(2*a*x + b)
    plt1, = plt.plot(x, y)
    plt2, = plt.plot(x, dydx)
    plt.plot([-1], [0], 'x')
    plt.plot([5], [12], 'x')
    plt.plot([10], [22], 'x')
    plt.grid()
    plt.legend((plt1, plt2), ('y', 'dy/dx'))
    plt.show()

# y = sin(x) = sin(2pi*f*t) = sin(w*t): x = 2pi*f*t
#           x     y
# 0deg      0     0    -> np.sin(0)
# 90deg     pi/2  1    -> np.sin(np.pi/2) ~ np.sin(1.57)
# 180deg    pi    0    -> np.sin(np.pi) ~ np.sin(3.14)
def plot_sin():
    x = np.arange(0.0, 10.0, 0.1)
    y = np.sin(x)
    f = 1
    t = x / 2 / np.pi / f
    y = np.sin(2 * np.pi * f * t)

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, y)
    axs[0].set(xlabel='x', ylabel='y')
    axs[0].grid()
    axs[1].plot(t, y)
    axs[1].set(xlabel='t (f = 1Hz)', ylabel='y')
    axs[1].grid()
    fig.suptitle('sine wave')
    plt.show()

# Taylor of sin(x) around a = pi/4
def plot_taylor_series():
    x = np.linspace(-2, 2, 100)
    y = np.sin(x)
    y0 = [1/np.sqrt(2)]* 100
    y1 = (x - np.pi/4)**1 / 1 / np.sqrt(2)
    y2 = -(x - np.pi/4)**2 / 2 / np.sqrt(2)
    y3 = -(x - np.pi/4)**3 / 6 / np.sqrt(2)
    y4 = (x - np.pi/4)**4 / 24 / np.sqrt(2)
    y5 = y0 + y1 + y2 + y3 + y4

    p, = plt.plot(x, y)
    p0, = plt.plot(x, y0, ':')
    p1, = plt.plot(x, y1, '-.')
    p2, = plt.plot(x, y2, '--')
    p3, = plt.plot(x, y3, ':')
    p4, = plt.plot(x, y4, '-.')
    p5, = plt.plot(x, y5, color = 'orange')
    plt.plot([np.pi/4], [np.sin(np.pi/4)], 'x', ms=12, markeredgewidth=3, color='black')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Taylor Series')
    plt.legend((p, p0, p1, p2, p3, p4, p5), ('sin(x)', '0th term', '1st term', '2nd term', '3rd term', '4th term', '5th term', 'sum'))
    plt.grid()
    plt.show()

# numeric integral estimate: Trapezoidal rule
# https://mathinsight.org/numerical_integration_refresher
# distribution function has no closed-form solution, hence start with density function
def plot_gaussian():
    def f(x):
        return np.exp(-(x-mean)**2/2/covar**2)/np.sqrt(2*np.pi*covar**2)

    mean = 0
    covar = 2
    x = np.linspace(-8, 8, 100)
    y1 = np.exp(-(x-mean)**2/2/covar**2)/np.sqrt(2*np.pi*covar**2)
    y2 = [0]
    val = 0
    # numeric integral estimate: Trapezoidal rule
    for j in range(1, len(x)):
        for i in range(j+1):
            if i == 0 or i == j:
                val = val + f(x[i])
            else:
                val = val + 2 * f(x[i])
        y2.append(val * (x[j]-x[0])/(j+1) / 2)
        val = 0

    plt.grid()
    p0, = plt.plot(x, y1)
    p1, = plt.plot(x, y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian')
    plt.legend((p0, p1), ('Probability density', 'Probability distribution'))
    plt.show()

def plot_multivariate_gaussian():
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 1.])
    Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    F = multivariate_normal(mu, Sigma)
    Z = F.pdf(pos)

    # Create a surface plot and projected filled contour plot under it.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
    #                 cmap=cm.viridis)

    #cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
    plt.contourf(X, Y, Z, zdir='z', alpha=0.5, cmap=plt.get_cmap('jet'))
    # Adjust the limits, ticks and view angle
    # ax.set_zlim(-0.15,0.2)
    # ax.set_zticks(np.linspace(0,0.2,5))
    # ax.view_init(27, -21)

    plt.show()

def plot_sigmoid():
    x = np.linspace(-8, 8, 100)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y)
    plt.grid()
    plt.xlabel('z')
    plt.ylabel('sigma(z)')
    plt.title('Sigmoid Function')
    plt.show()

# used to generate fake Pokemon CP value
# y: actual CP after evolution
# x: CP before evolution
def fake_pokomon_cp_function():
    A = 125
    phase = 0.25
    w1 = 0.9
    w2 = 0.05
    w3 = 0.05
    x = np.arange(10.0, 200.0, 1)
    height = np.random.randint(low=10, high=100, size=190)
    weight = np.random.randint(low=10, high=100, size=190)
    y = w1 * A * np.sin(x/100 + phase) + w2 * height + w3 * weight
    # plot
    #plt.plot(x, y, 'o')
    #plt.grid()
    #plt.show()
    # y: conver to int, then back to float (.0)
    return x, y.astype('int').astype('float'), height.astype('float'), weight.astype('float')

# randomly pick 10 pokemons from our fake function
def sample_pokemon():
    x, y, h, w = fake_pokomon_cp_function()
    sample_x = list()
    sample_y = list()
    sample_h = list()
    sample_w = list()
    index_list = np.random.randint(low=0, high=190, size=10)
    index_list.sort()
    print('index: {}'.format(index_list.tolist()))
    for i in index_list: 
        sample_x.append(x[i])
        sample_y.append(y[i])
        sample_h.append(h[i])
        sample_w.append(w[i])
    print('old cp: {}'.format(sample_x))
    print('new cp: {}'.format(sample_y))
    print('height: {}'.format(sample_h))
    print('weight: {}'.format(sample_w))
    plt.plot(sample_x, sample_y, 'o')
    plt.grid()
    plt.show()

# y = f(x) = 1 + sin(2 * pi * t)
def plot_1_plus_sin_2_pi_t():
    t = np.arange(0.0, 2.0, 0.01)
    f = 1 + np.sin(2 * np.pi * t)
    plt.plot(t, f)
    plt.xlabel('time (s)')
    plt.ylabel('voltage (mV)')
    plt.title('test plot')
    plt.grid()
    plt.show()

# z = sin(x^2 + y^2) / (x^2 + y^2)
def contourf():
    x = y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)     # sparse can be omitted
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    plt.contourf(x, y, z)
    plt.show()

# timing diagram: 1 if 2 < t < 5, or 8 < t < 9, 0 otherwise
def timing_diagram():
    t = np.linspace(0, 10, 1000)
    p1 = np.logical_or(np.logical_and(t > 2, t < 5), np.logical_and(t > 8, t < 9))
    p2 = np.logical_or(np.logical_and(t > 1.5, t < 5.5), np.logical_and(t > 7.8, t < 8.8))
    fig, axs = plt.subplots(2, 1)
    # plot
    axs[0].plot(t, p1)
    axs[1].plot(t, p2)
    # setup
    axs[0].grid()
    axs[1].grid()
    axs[0].set(ylabel='p1')
    axs[1].set(xlabel='time', ylabel='p2')
    # show
    plt.show()

def multiple_plots():
    t = np.arange(0.0, 2.0, 0.01)
    s1 = np.sin(2*np.pi*t)
    s2 = np.sin(4*np.pi*t)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t, s1)
    plt.subplot(212)
    plt.plot(t, 2*s1)
    plt.figure(2)
    plt.plot(t, s2)
    plt.show()

def main():
    #print(plot())
    #plot_ax2_bx_c()
    #plot_taylor_series()
    #plot_gaussian()
    #plot_sigmoid()
    #plot_multivariate_gaussian()
    #test_plot()
    multiple_plots()
    
if __name__ == '__main__':
    main()