import numpy as np
import matplotlib.pyplot as plt

def plot():
    n = 8
    #y = np.zeros(n)
    x = np.linspace(0, 10, n)
    y = x
    plt.plot(x, y, 'o')
    plt.ylim([-0.5, 10])
    plt.show()

def contourf():
    x = y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)     # sparse can be omitted
    z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    plt.contourf(x, y, z)
    plt.show()

def main():
    print(plot())

if __name__ == '__main__':
    main()