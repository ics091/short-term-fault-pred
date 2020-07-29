import pandas as pd


def path(time):
    h_data_path = './' + time + '/h/h'
    nh_data_path = './' + time + '/nh/nh'
    return h_data_path, nh_data_path


def siphon(step, data, k, cb_data, label_list, label):
    for i in range(step):
        box = []
        p = i
        while p < (step * k):
            box.append(
                data.iloc[[p],
                          [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist()[0])
            p += step
        cb_data.append(box)
        label_list.append(label)


# 抽取
def load_HNH_siphon(l1, l2, time):
    p1, p2 = path(time)
    x = 1
    y = 1
    cb_data = []
    label = []
    while x <= l1:
        try:
            p = p1 + str(x) + '.csv'
            data = pd.read_csv(p, header=None)
            k = 30
            l = len(data)
            if k <= l < 2 * k:
                for i in range(1):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
            elif 2 * k <= l < 3 * k:
                siphon(2, data, k, cb_data, label, 1)
            elif 3 * k <= l < 4 * k:
                siphon(3, data, k, cb_data, label, 1)
            elif 4 * k <= l < 5 * k:
                siphon(4, data, k, cb_data, label, 1)
            elif 5 * k <= l < 6 * k:
                siphon(5, data, k, cb_data, label, 1)
            elif 6 * k <= l:
                siphon(6, data, k, cb_data, label, 1)
        except:
            pass
        x += 1

    # print(len(label))

    while y <= l2:
        try:
            p = p2 + str(y) + '.csv'
            data = pd.read_csv(p, header=None)
            # print(len(data))
            k = 30
            l = len(data)
            if k <= l < 2 * k:
                for i in range(1):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
            elif 2 * k <= l < 3 * k:
                siphon(2, data, k, cb_data, label, 0)
            elif 3 * k <= l < 4 * k:
                siphon(3, data, k, cb_data, label, 0)
            elif 4 * k <= l < 5 * k:
                siphon(4, data, k, cb_data, label, 0)
            elif 5 * k <= l < 6 * k:
                siphon(5, data, k, cb_data, label, 0)
            elif 6 * k <= l:
                siphon(6, data, k, cb_data, label, 0)
        except:
            pass
        y += 1
    return cb_data, label


# 切片
def load_HNH(l1, l2, time):
    p1, p2 = path(time)
    x = 1
    y = 1
    cb_data = []
    label = []
    while x <= l1:
        try:
            p = p1 + str(x) + '.csv'
            data = pd.read_csv(p, header=None)
            k = 30
            l = len(data)
            if k <= l < 2 * k:
                for i in range(1):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
            elif 2 * k <= l < 3 * k:
                for i in range(2):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
            elif 3 * k <= l < 4 * k:
                for i in range(3):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
            elif 4 * k <= l < 5 * k:
                for i in range(4):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
            elif 5 * k <= l < 6 * k:
                for i in range(5):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
            elif 6 * k <= l:
                for i in range(6):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(1)
        except:
            pass
        x += 1

    # print(len(label))

    while y <= l2:
        try:
            p = p2 + str(y) + '.csv'
            data = pd.read_csv(p, header=None)
            # print(len(data))
            k = 30
            l = len(data)
            if k <= l < 2 * k:
                for i in range(1):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
            elif 2 * k <= l < 3 * k:
                for i in range(2):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
            elif 3 * k <= l < 4 * k:
                for i in range(3):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
            elif 4 * k <= l < 5 * k:
                for i in range(4):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
            elif 5 * k <= l < 6 * k:
                for i in range(5):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
            elif 6 * k <= l:
                for i in range(6):
                    cb_data.append(data.iloc[
                        i * k:(i + 1) * k,
                        [1, 2, 3, 4, 5, 8, 10, 13, 14]].values.tolist())
                    label.append(0)
        except:
            pass
        y += 1
    return cb_data, label
