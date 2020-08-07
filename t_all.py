import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import random
from matplotlib import pyplot as plt
from data_loader import load_HNH
from data_loader import load_HNH_siphon

SEED = 1024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_data():
    time_list = [
        '2018-11', '2018-12', '2019-01', '2019-08', '2019-09', '2019-10'
    ]
    data_1811, label_1811 = load_HNH_siphon(52, 117, time_list[0])
    data_1812, label_1812 = load_HNH_siphon(40, 69, time_list[1])
    data_1901, label_1901 = load_HNH_siphon(10, 32, time_list[2])
    data_1908, label_1908 = load_HNH_siphon(55, 98, time_list[3])
    data_1909, label_1909 = load_HNH_siphon(25, 62, time_list[4])
    data_1910, label_1910 = load_HNH_siphon(45, 90, time_list[5])

    X = []
    Y = []

    for x, y in zip(data_1811, label_1811):
        X.append(x)
        Y.append(y)

    for x, y in zip(data_1812, label_1812):
        X.append(x)
        Y.append(y)

    for x, y in zip(data_1901, label_1901):
        X.append(x)
        Y.append(y)

    for x, y in zip(data_1908, label_1908):
        X.append(x)
        Y.append(y)

    for x, y in zip(data_1909, label_1909):
        X.append(x)
        Y.append(y)

    for x, y in zip(data_1910, label_1910):
        X.append(x)
        Y.append(y)

    return X, Y


def pre_data(X, y):
    X_train, X_vail_test, y_train, y_vail_test = train_test_split(
        X, y, test_size=0.4)
    X_vail, X_test, y_vail, y_test = train_test_split(X_vail_test,
                                                      y_vail_test,
                                                      test_size=0.5)
    X_train = X_train.reshape(-1, 9)
    X_vail = X_vail.reshape(-1, 9)
    X_test = X_test.reshape(-1, 9)
    mean_std_scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = mean_std_scaler.transform(X_train).reshape(-1, 30, 9)
    X_vail = mean_std_scaler.transform(X_vail).reshape(-1, 30, 9)
    X_test = mean_std_scaler.transform(X_test).reshape(-1, 30, 9)

    X_train = torch.from_numpy(X_train).float()
    X_vail = torch.from_numpy(X_vail).float()
    X_test = torch.from_numpy(X_test).float()

    y_train = torch.from_numpy(y_train).float()
    y_vail = torch.from_numpy(y_vail).float()
    y_test = torch.from_numpy(y_test).float()
    return X_train, X_vail, X_test, y_train, y_vail, y_test


# pred result to one zero
def to_one_zero(X):
    for i in range(len(X)):
        if X[i][0] >= 0.5:
            X[i][0] = 1
        else:
            X[i][0] = 0
    X = X.reshape(-1, )
    return X


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv1d(INTPUT_SIZE, 8, 2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_size=8,
            dropout=0.2,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
        )
        self.dense = nn.Sequential(nn.Dropout(0.5), nn.Linear(16, 6),
                                   nn.ReLU(), nn.Linear(6, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.cnn(x)
        out = out.permute(0, 2, 1)
        h0 = c0 = torch.randn(2, x.shape[0], 16)
        r_out, _ = self.rnn(out, (h0, c0))
        r_out = r_out[:, -1, :]
        out = self.dense(r_out)
        return out


BATCH_SIZE = 256
TIME_STEP = 30
INTPUT_SIZE = 9
LR = 0.001

if __name__ == '__main__':
    X, y = load_data()
    X = np.array(X)  # (1181, 30, 9)
    y = np.array(y)
    print(X.shape)
    print(y.shape)

    X_train, X_vail, X_test, y_train, y_vail, y_test = pre_data(X, y)

    print(len(X_train))
    print(len(X_vail))
    print(len(X_test))

    dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    lstm = LSTM()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
    loss_func = nn.BCEWithLogitsLoss()  # 在BCELoss的基础上添加sigmoid
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           patience=10,
                                                           verbose=True)

    epoch_rec = []
    train_loss_rec = []
    test_loss_rec = []
    AUC_rec = []
    accu_rec = []
    sum_loss = 0.0
    sum_step = 1

    for epoch in range(200):
        for step, (b_x, b_y) in enumerate(loader):

            b_x = b_x.view(-1, 30, 9)
            output = lstm(b_x)
            b_y = b_y.numpy().reshape(-1, 1)
            b_y = torch.from_numpy(b_y)
            loss = loss_func(output, b_y)
            sum_loss = sum_loss + loss
            sum_step = sum_step + 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                train_loss_rec.append(sum_loss / sum_step)
                sum_loss = 0.0
                sum_step = 0
                lstm = lstm.eval()
                with torch.no_grad():
                    v_output = lstm(X_vail.view(-1, 30, 9))
                    y_vail = y_vail.numpy().reshape(-1, 1)
                    y_vail = torch.from_numpy(y_vail)
                    v_loss = loss_func(v_output, y_vail)
                    true_vy = y_vail.numpy().reshape(-1, )
                    pred_vy = to_one_zero(torch.sigmoid(v_output).numpy())
                    print(
                        'epoch', epoch + 1, '|step', step,
                        '\n|loss: %.5f ' % v_loss.data.numpy(), '|AUC: %.5f ' %
                        roc_auc_score(y_vail, torch.sigmoid(v_output)),
                        '|accu: %.5f ' % accuracy_score(pred_vy, true_vy))
                    # print('pred_y:',pred_y)
                    # print('true_y:',true_y)

                    epoch_rec.append(epoch)
                    AUC_rec.append(
                        roc_auc_score(y_vail, torch.sigmoid(v_output)))
                    accu_rec.append(accuracy_score(pred_vy, true_vy))
                lstm.train()

        lstm.eval()
        with torch.no_grad():
            t_out = lstm(X_test.view(-1, 30, 9))
            y_test = y_test.numpy().reshape(-1, 1)
            y_test = torch.from_numpy(y_test)
            t_loss = loss_func(t_out, y_test)
            test_loss_rec.append(t_loss)
            t_true_y = y_test.numpy().reshape(-1, )
            t_pred_y = to_one_zero(torch.sigmoid(t_out).numpy())
            print('|loss: %.5f ' % t_loss.data.numpy(),
                  '|AUC: %.5f ' % roc_auc_score(y_test, torch.sigmoid(t_out)),
                  '|accu: %.5f ' % accuracy_score(t_pred_y, t_true_y))
        lstm.train()

    epoch_rec = np.array(epoch_rec)
    train_loss_rec = np.array(train_loss_rec)
    test_loss_rec = np.array(test_loss_rec)
    AUC_rec = np.array(AUC_rec)
    accu_rec = np.array(accu_rec)
    plt.plot(epoch_rec, train_loss_rec, label='train-loss')
    plt.plot(epoch_rec, test_loss_rec, label='test-loss')
    # plt.plot(epoch_rec, AUC_rec, label='AUC')
    # plt.plot(epoch_rec,accu_rec,label='accu')
    plt.xlabel('epoch')
    plt.ylabel('LOSS')
    plt.legend()
    plt.show()
