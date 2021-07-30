from sklearn.datasets import load_digits
from com.utility import *

if __name__ == '__main__':
    digits = load_digits()
    X, Y = digits.data, digits.target

    print(X, '\n', Y.shape)

    x, y = loadData()
    x = np.array(x)
    y = np.array(y.values)

    y = y.flatten()

    print(x, '\n', y.shape)
