import numpy as np
from dtw import dtw

l2_norm = lambda x, y: (x - y)**2


def mean(arr):
    return float(sum(arr)) / max(len(arr), 1)


def Medoid(D):
    minSS = float("inf")
    for S1 in D:
        tmpSS = 0
        for S2 in D:
            tmpSS += (dtw(S1, S2, dist=l2_norm)[0])**2
        if tmpSS < minSS:
            medoid = S1
            minSS = tmpSS
    return medoid


def Wtd_medoid(D, W):
    minSS = float("inf")
    for S1 in D:
        tmpSS = 0
        for S2, w in zip(D, W):
            tmpSS += w * ((dtw(S1, S2, dist=l2_norm)[0])**2)
        if tmpSS < minSS:
            medoid = S1
            minSS = tmpSS
    return medoid


def DTW_multiple_alignment(Sref, S):
    dist, cost_matrix, cost, path = dtw(Sref, S, dist=l2_norm)
    L = len(Sref)
    alignment = []
    for l in range(L):
        alignment.append([])
    i = cost.shape[0] - 1
    j = cost.shape[1] - 1
    while (i >= 0) and (j >= 0):
        alignment[i].append(S[j])
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            score = min(cost[i - 1][j - 1], cost[i][j - 1], cost[i - 1][j])
            if score == cost[i - 1][j - 1]:
                i = i - 1
                j = j - 1
            elif score == cost[i - 1][j]:
                i = i - 1
            else:
                j = j - 1
    return alignment


def DTW_multiple_alignment_p(Sref, S):
    dist, cost_matrix, acc_cost_matrix, path = dtw(Sref, S, dist=l2_norm)
    L = len(Sref)
    alignment = []
    for l in range(L):
        alignment.append([])
    for m, n in zip(path[0], path[1]):
        alignment[m].append(S[n])
    return alignment



def DBA_update(Tinit, D):
    T = []
    alignment = []
    L = len(Tinit)
    for l in range(L):
        alignment.append([])
        T.append([])
    for S in D:
        alignment_for_S = DTW_multiple_alignment_p(Tinit, S)
        for i in range(L):
            alignment[i].append(alignment_for_S[i])
    # average
    for j in range(L):
        for A in alignment[j]:
            for a in A:
                T[j].append(a)
        T[j] = mean(T[j])
    return T


def Wtd_DBA_update(Tinit, D, W):
    T = []
    alignment = []
    sumWeights = []
    L = len(Tinit)
    for l in range(L):
        alignment.append([])
        T.append([])
        sumWeights.append(0)
    for S, w in zip(D, W):
        alignment_for_S = DTW_multiple_alignment_p(Tinit, S)
        for i in range(L):
            alignment[i].append([s * w for s in alignment_for_S[i]])
            sumWeights[i] += w
    # average
    for j in range(L):
        for A in alignment[j]:
            for a in A:
                T[j].append(a)
        T[j] = float(sum(T[j])) / sumWeights[j]
    return T


def DBA(D, I):
    # I means times of update
    T = Medoid(D)
    for i in range(I):
        T = DBA_update(T, D)
        print(T)
    return T


def Wtd_DBA(D, W, I):
    # I means times of update
    T = Wtd_medoid(D, W)
    for i in range(I):
        T = Wtd_DBA_update(T, D, W)
        print(T)
    return (T)


if __name__ == '__main__':
    D = [
        np.array([1, 3, 6, 6, 5, 3, 2]),
        np.array([0, 2, 5, 4, 2, 1, 1, 0]),
        np.array([0, 0, 1, 3, 0, 1, 1]),
        np.array([1, 1, 3, 3, 4, 4, 2, 0]),
        np.array([0, 0, 1, 3, 2, 1, 0])
    ]

    # W = np.random.rand(5)
    # print(W)
    # print(sum(W))
    W = [1, 0.3, 0.4, 0.766, 0.2]
    # medoid_ = Medoid(D)
    # print(medoid_)
    # print(DTW_multiple_alignment(medoid_, D[0]))
    # T = DBA(D, 5)
    # m = Medoid(D)
    # w_m = Wtd_medoid(D, W)
    # print(m)
    # print(w_m)
    # Wtd_DBA(D, W, 5)
    # print(DTW_multiple_alignment(D[0], D[1]))
    # print(DTW_multiple_alignment_p(D[0], D[1]))
    Wtd_DBA(D, W, 5)
