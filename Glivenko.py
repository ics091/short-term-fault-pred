import numpy as np
from matplotlib import pyplot as plt

# X = [12, 23, 43, 12, 45, 32, 8, 39, 10, 41]
# X = [0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
X = [1, 3, 1, 2, 4]
Y = [14, 13, 17, 11, 12]


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


# R, P = distribute(X, False)
# print(R)
# print(P)
# plotDistriFx(R, P, 0.1)


def distribute2d(X, Y, Avg):
    X = np.array(X)
    Y = np.array(Y)
    length = min(len(X), len(Y))
    coord = np.empty([length, 2], dtype=float)
    if Avg:
        RX = np.sort(np.array(X))
        RY = np.sort(np.array(Y))
        RX = np.linspace(RX[0],
                         RX[length - 1],
                         length,
                         endpoint=True,
                         dtype=float)
        RY = np.linspace(RY[0],
                         RY[length - 1],
                         length,
                         endpoint=True,
                         dtype=float)
    else:
        RX = X
        RY = Y
    for i in range(length):
        coord[i][0] = RX[i]
        coord[i][1] = RY[i]
    coord = np.sort(coord, axis=0)
    n = len(coord)
    for i in range(1, n):
        if (coord[i][0] == coord[i - 1][0] or coord[i][1] == coord[i - 1][1]):
            coord[i - 1] = coord[i]
    coord = np.unique(coord, axis=0)
    n = len(coord)
    P = np.empty(n, dtype=float)
    for i in range(n):
        sum_N = 0
        for t in range(length):
            if (X[t] <= coord[i][0] and Y[t] <= coord[i][1]):
                sum_N += 1
        P[i] = sum_N / length
    return coord, P


def plotXY(X, Y):
    plt.plot(X, Y, '.')
    plt.grid()
    plt.show()


C, P = distribute2d(X, Y, False)
print(C)
print(P)
plotXY(X, Y)
