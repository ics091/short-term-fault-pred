import numpy as np
from DBA import Medoid, Wtd_medoid, Weight_Assign, Wtd_DBA

Data = [[[100, 50, 1, 0.1], 
         [98, 45, 0.9, 0.2], 
         [78, 30, 1.1, 0.15]],
        [[89, 48, 1.3, 0.09], 
         [67, 39, 1, 0.1], 
         [56, 45, 1.4, 0.3]],
        [[81, 41, 1.11, 0.11], 
         [78, 39, 1.2, 0.14], 
         [66, 36, 1.21, 0.13],
         [66, 33, 1, 0.11]], 
         [[91, 46, 0.98, 0.21], 
          [78, 34, 1.12, 0.16]],
        [[71, 50, 1.31, 0.2], 
         [69, 41, 1.2, 0.15], 
         [62, 32, 1.71, 0.12],
         [56, 31, 1.4, 0.09]]]


def DBAugment(Data, r, cycle):
    L = len(Data[0][0])
    X = []
    Generate = []
    for l in range(L):
        X.append([])
    for i in range(L):
        for S in Data:
            # print([c[i] for c in S])
            X[i].append(np.array([c[i] for c in S]))
    # random get a init
    index = np.random.randint(0, len(Data), size=1)[0]
    for D in X:
        Tinit = D[index]
        # assign weight
        W = Weight_Assign(Tinit, D, r)
        # update T
        T = Wtd_DBA(D, W, cycle)
        # print(T)
        Generate.append(T)
    minLen = float("inf")
    for g in Generate:
        if len(g) < minLen:
            minLen = len(g)
    for i in range(len(Generate)):
        Generate[i] = Generate[i][:minLen]
    Generate = np.transpose(np.matrix(Generate))
    print(Generate)
    return Generate


if __name__ == '__main__':
    DBAugment(Data, 0.5, 5)
