import numpy as np
from matplotlib import pyplot as plt

X = [12, 23, 43, 12, 45, 32, 8, 39, 10, 41]


def distribute(X, Avg):
    X = np.sort(np.array(X))
    print(X)
    length = len(X)
    if Avg:
        R = np.linspace(X[0],
                        X[length - 1],
                        length,
                        endpoint=True,
                        dtype=float)
    else:
        R = np.unique(X)
    n = len(R)
    P = np.empty(n, dtype=float)
    for i in range(n):
        sum_N = 0
        for x in X:
            if (x <= R[i]):
                sum_N += 1
            else:
                break
        P[i] = sum_N / length
    return R, P


def plotDistriFx(R, P, precision):
    length = len(R)
    x_point = np.arange(R[0] - 10 * precision,
                        R[length - 1] + 10 * precision,
                        precision,
                        dtype=float)
    y_P = np.empty(len(x_point), dtype=float)
    for i in range(len(x_point)):
        A = np.where(R > x_point[i])[0]
        if any(A):
            index = A[0] - 1
            if index == -1:
                y_P[i] = 0
            else:
                y_P[i] = P[index]
        else:
            y_P[i] = 1
    plt.plot(x_point, y_P, '.')
    plt.show()


R, P = distribute(X, False)
# print(R)
# print(P)
plotDistriFx(R, P, 0.1)
