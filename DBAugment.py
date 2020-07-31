import numpy as np
from DBA import Weight_Assign, Wtd_DBA
import pandas as pd

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


def path(time):
    h_data_path = './' + time + '/h/h'
    nh_data_path = './' + time + '/nh/nh'
    return h_data_path, nh_data_path


def loadOriginHNH(l1, l2, time):
    p1, p2 = path(time)
    x = 1
    y = 1
    cb_data1 = []
    cb_data2 = []
    while x <= l1:
        try:
            p = p1 + str(x) + '.csv'
            data = pd.read_csv(p, header=None)
            if len(data) >= 30:
                cb_data1.append(data.values.tolist)
            else:
                pass
        except:
            pass
        x += 1
    while y <= l2:
        try:
            p = p2 + str(y) + '.csv'
            data = pd.read_csv(p, header=None)
            if len(data) >= 30:
                cb_data2.append(data.values.tolist)
            else:
                pass
        except:
            pass
        y += 1
    
    return cb_data1, cb_data2

# 
def AugmentHNH(Data, r, cycle, N, path, type):
    for n in range(N):
        GenTS = DBAugment(Data, r, cycle)
        GenTS = pd.DataFrame(data=np.array(GenTS))
        if type == 0:
            csv_path = path + '/nh/nh' + str(n + 1) + '.csv'
        elif type == 1:
            csv_path = path + '/h/h' + str(n + 1) + '.csv'
        else:
            pass
        GenTS.to_csv(csv_path, index=False, header=False)


if __name__ == '__main__':
    
