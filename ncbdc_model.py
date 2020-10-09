import os
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss



cdtype = {
    # 'yr_modahrmn': str,
    # 'message_type': str,
    # 'speed': float,
    # 'vehicle_state': int,
    # 'charging_status': int,
    # 'mode': int,
    # 'total_volt': float,
    # 'total_current': float,
    # 'mileage': float,
    # 'standard_soc': int,
    'cell_volt_list': str,
    # 'max_cell_volt': float,
    # 'max_volt_cell_id': int,
    # 'min_cell_volt': float,
    # 'min_cell_volt_id': int,
    # 'max_temp': int,
    # 'max_temp_probe_id': int,
    # 'min_temp': int,
    # 'min_temp_probe_id': int,
    # 'max_alarm_lvl': int,
    # 'bat_fault_list': object,
    # 'isulate_r': int,
    # 'dcdc_stat': int,
    # 'sing_temp_num': int,
    # 'sing_volt_num': int,
    'cell_temp_list': str,
    'gear': int,
    # 'max_volt_num': int,
    # 'min_volt_num': int,
    # 'max_temp_num': int,
    # 'min_temp_num': int,
    'alarm_info': str
}
feature_index = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,21,22,26]


root_path = '/data/04/train/'
# battery type
class BaType(Enum):
    SAN = 'san' # 三元材料
    LIN = 'lin' # 磷酸铁锂
    MEN = 'men' # 锰酸锂
    
# data type
class DataType(Enum):
    NOM = 'normal' # 正常数据
    BFO = 'before' # 报警前
    CTN = 'continue' # 持续报警
    
SAN_NOM = root_path + BaType['SAN'].value + '/' + DataType['NOM'].value + '/'
SAN_BFO = root_path + BaType['SAN'].value + '/' + DataType['BFO'].value + '/'
SAN_CTN = root_path + BaType['SAN'].value + '/' + DataType['CTN'].value + '/'

LIN_NOM = root_path + BaType['LIN'].value + '/' + DataType['NOM'].value + '/'
LIN_BFO = root_path + BaType['LIN'].value + '/' + DataType['BFO'].value + '/'
LIN_CTN = root_path + BaType['LIN'].value + '/' + DataType['CTN'].value + '/'

MEN_NOM = root_path + BaType['MEN'].value + '/' + DataType['NOM'].value + '/'
MEN_BFO = root_path + BaType['MEN'].value + '/' + DataType['BFO'].value + '/'
MEN_CTN = root_path + BaType['MEN'].value + '/' + DataType['CTN'].value + '/'

NOM = [SAN_NOM, LIN_NOM, MEN_NOM]
BFO = [SAN_BFO, LIN_BFO, MEN_BFO]
CTN = [SAN_CTN, LIN_CTN, MEN_CTN]

def process(type):
    cbn_data = []
    for path in type:
        csv_files = os.listdir(path)
        for f in csv_files:
            df = pd.read_csv(path + str(f), header=0)
            df = df.iloc[:, feature_index].fillna(0)
            data = df.values.tolist()
            cbn_data.append(data)
    return cbn_data

def get_data():
    nom_data = np.array(process(NOM))
    bfo_data = np.array(process(BFO))
    ctn_data = np.array(process(CTN))
    # 正常数据 lable=0
    nom_label = np.zeros(len(nom_data))
    # 报警前数据 label=1
    bfo_label = np.ones(len(bfo_data))
    # 持续报警数据 label=2
    ctn_label = np.ones(len(ctn_data))*2
    
    X = np.concatenate((nom_data, bfo_data), axis=0)
    X = np.concatenate((X, ctn_data), axis=0)
    y = np.concatenate((nom_label, bfo_label), axis=0)
    y = np.concatenate((y, ctn_label), axis=0)
    print(X.shape)
    print(y.shape)
    print('data prepared')
    # (4935, 6, 19)
    # (4935,)
    return X, y

ft_size = len(feature_index) # 输入特征个数
time_step = 6 # 一个样本的时间跨度1min,对应6条连续的数据

# 数据预处理
def pre_data(X, y):
    
    # split: train valid test 6:2:2
    X_train, X_vail_test, y_train, y_vail_test = train_test_split(X, y, test_size=0.4)
    X_vail, X_test, y_vail, y_test = train_test_split(X_vail_test,
                                                      y_vail_test,
                                                      test_size=0.5)
    # nomalization
    
    X_train = X_train.reshape(-1, ft_size)
    X_vail = X_vail.reshape(-1, ft_size)
    X_test = X_test.reshape(-1, ft_size)
    
    mean_std_scaler = preprocessing.StandardScaler().fit(X_train)
    
    X_train = mean_std_scaler.transform(X_train).reshape(-1, time_step, ft_size)
    X_vail = mean_std_scaler.transform(X_vail).reshape(-1, time_step, ft_size)
    X_test = mean_std_scaler.transform(X_test).reshape(-1, time_step, ft_size)
    
    X_train = torch.from_numpy(X_train).float()
    X_vail = torch.from_numpy(X_vail).float()
    X_test = torch.from_numpy(X_test).float()
    
    y_train = torch.from_numpy(y_train).long()
    y_vail = torch.from_numpy(y_vail).long()
    y_test = torch.from_numpy(y_test).long()
    
    print('train: ' + str(len(X_train)) + ' vaild: ' + str(len(X_vail)) + ' test: ' + str(len(X_test)))
    print('preprocess data finish')
    return X_train, X_vail, X_test, y_train, y_vail, y_test


# model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size = ft_size,
            hidden_size = 32,
            num_layers = 3,
            batch_first = True,
        )
        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        
    def forward(self, x):
        h0 = torch.randn(3, x.shape[0], 32).cuda()
        c0 = torch.randn(3, x.shape[0], 32).cuda() 
        r_out, h_n = self.rnn(x, (h0, c0))
        out = h_n[0][1]
        out = self.dense(out)
        return out
    

# model parameter
BATCH_SIZE = 4096
LR = 0.01
EPOCH = 1000

X, y = get_data()
X_train, X_vail, X_test, y_train, y_vail, y_test = pre_data(X, y)

if __name__ == '__main__':
    data_set = Data.TensorDataset(X_train, y_train)
    
    train_loader = Data.DataLoader(
        dataset = data_set,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2,
    )
    
    model = LSTM().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data
            model.train()
            b_x = b_x.view(-1, time_step, ft_size)
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            
            output = model(b_x)
            
            loss = loss_func(output, b_y.squeeze())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # valid
            if step % 20 == 0:
                model.eval()
                with torch.no_grad():
                    vail_output = model(X_vail.view(-1, time_step, ft_size).cuda())
                    y_vail = y_vail.cuda()
                    vail_loss = loss_func(vail_output, y_vail.squeeze())
                    
                    pred_y = torch.max(vail_output, 1)[1].data.cpu().numpy() # 返回每行最大值的索引，即预测的类型
                    accuracy = accuracy_score(y_vail.cpu().numpy(), pred_y)
                    print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)

    # test
    model.eval()
    test_x = X_test.view(-1, time_step, ft_size).cuda()
    output = model(test_x)
    pred_y = torch.max(output, 1)[1].data.cpu().numpy()
    ture_y = y_test.numpy()
    accuracy = accuracy_score(ture_y, pred_y)
    print(pred_y)
    print(accuracy)