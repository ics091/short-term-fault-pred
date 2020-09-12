import pandas as pd
import math
from matplotlib import pyplot as plt
import numpy as np


def abnml_mileage(value):
    if (value == 4294967294 or value == 4294967295 or value < 0
            or value > 9999999):
        return True
    else:
        return False


def analysis(path):
    try:
        data = pd.read_csv(
            path,
            header=0,
            dtype={
                'yr_modahrmn': str,
                'message_type': str,
                'speed': float,
                # 'vehicle_state': int,
                # 'charging_status': int,
                # 'mode': int,
                # 'total_volt': float,
                # 'total_current': float,
                # 'mileage': float,
                # 'standard_soc': int,
                # 'cell_volt_list': str,
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
                # 'cell_temp_list': str,
                # 'gear': int,
                # 'max_volt_num': int,
                # 'min_volt_num': int,
                # 'max_temp_num': int,
                # 'min_temp_num': int,
                # 'alarm_info': str
            })
    except Exception as e:
        print(str(e))
    _speed = np.array(data['speed'].values.tolist())
    _Vstatus = np.array(data['vehicle_state'].values.tolist())
    _Cstatus = np.array(data['charging_status'].values.tolist())
    _Tcurrent = np.array(data['total_current'].values.tolist())
    _Tcurrent = _Tcurrent / 100
    _Tvolt = np.array(data['total_volt'].values.tolist())
    _Tvolt = _Tvolt / 100
    _mile = np.array(data['mileage'].values.tolist())
    _soc = np.array(data['standard_soc'].values.tolist())
    _soc = _soc / 100
    _maxvolt = np.array(data['max_cell_volt'].values.tolist())
    _minvolt = np.array(data['min_cell_volt'].values.tolist())
    index = np.arange(0, len(data), 1)
    # plt.plot(index, _speed, label='speed')
    # plt.plot(index, _Vstatus, label='vehicle stt')
    plt.plot(index, _Cstatus, label='charge stt')
    # plt.plot(index, _soc, label='soc')
    plt.plot(index, _Tvolt, label='volt')
    plt.plot(index, _Tcurrent, label='current')
    # plt.plot(index, _mile, label='mile')
    # plt.plot(index, _maxvolt, label='max_volt')
    # plt.plot(index, _minvolt, label='min_volt')
    plt.legend()
    plt.grid()
    plt.show()
    # for s, vtt, ctt, c, v, m, soc in zip(_speed, _Vstatus, _Cstatus, _Tcurrent,
    #                                      _Tvolt, _mile, _soc):
    # vehicle_state
    # if (s == 0.0 and ctt == 3 and c <= 0):
    #    print('Vstatus= ' + str(vtt) + ' _Tcurrent= ' + str(c) +
    #          ' _Tvolt = ' + str(v))
    # else:
    #     pass
    # if (ctt == 3 and vtt == 2):
    #     print('speed= ' + str(s) + ' Vstatus= ' + str(vtt) +
    #           ' _Tcurrent= ' + str(c))


def process_sigle(path):
    try:
        data = pd.read_csv(
            path,
            header=0,
            dtype={
                'yr_modahrmn': str,
                'message_type': str,
                'speed': float,
                # 'vehicle_state': int,
                # 'charging_status': int,
                # 'mode': int,
                # 'total_volt': float,
                # 'total_current': float,
                # 'mileage': float,
                # 'standard_soc': int,
                # 'cell_volt_list': str,
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
                # 'cell_temp_list': str,
                # 'gear': int,
                # 'max_volt_num': int,
                # 'min_volt_num': int,
                # 'max_temp_num': int,
                # 'min_temp_num': int,
                # 'alarm_info': str
            })
    except Exception as e:
        print(str(e))
    for c in data.columns:
        # 获取列（数据项）及缺失值情况
        column_data = data[c]
        is_NaN = column_data.isna().any()
        if is_NaN:
            nan_list = column_data.index[np.where(
                np.isnan(column_data))[0]].values.tolist()
            co_list = []
            fNaN_list = []
            for d in nan_list:
                co_list.append(d)
                if d + 1 not in nan_list:
                    fNaN_list.append([co_list[0], co_list[-1]])
                    co_list = []
        is_empty = column_data.dropna().empty
        column_data = column_data.values.tolist()
        if is_empty:
            # 处理整个数据项为空的情况
            pass
        else:
            if (c == 'yr_modahrmn' or c == 'message_type'):
                pass
            elif (c == 'speed'):
                for i in range(len(column_data)):
                    value = column_data[i]
                    if (value == 254 or value == 255 or value < 0
                            or value > 2200):
                        print("index: " + str(i) + "error vaule " + value)
                    elif (math.isnan(value)):
                        print("index: " + str(i) + " NaN value")
            elif (c == 'vehicle_state'):
                for i in range(len(column_data)):
                    value = column_data[i]
                    if (value == 254 or value == 255):
                        print("index: " + str(i) + "error vaule " + value)
                    elif (math.isnan(value)):
                        _speed = data['speed'][i]
                        _Cstatus = data['charging_status'][i]
                        _crt = data['total_current'][i]
                        if (_speed == 0 and _Cstatus == 1):
                            # 车速为 0且充电状态为 1（停车充电）=>车辆状态为熄火 2
                            column_data[i] = 2
                        elif (_speed > 0 and _Cstatus == 3):
                            # 车速大于 0且充电状态为 3（未充电）=>车辆状态启动 1
                            column_data[i] = 1
                        elif (_speed == 0 and _Cstatus == 3):
                            # 速度为 0 且充电状态为 3（未充电）=>
                            # 可能车辆状态启动 1 熄火 2
                            # 如果总电流 > 0 令为启动状态
                            if _crt > 0:
                                column_data[i] = 1
                            else:
                                column_data[i] = 2
                        else:
                            # 其他情况都列为未知状态 3
                            # 比如会 车速 充电状态 总电流 为空值的情况
                            column_data[i] = 3
            elif (c == 'charging_status'):
                _soc = data['standard_soc'][i]
                # 其实可以考虑取当前时间点之后一段时间soc的变化
                _speed = data['speed'][i]
                _Vstatus = data['vehicle_state'][i]
                _crt = data['total_current'][i]
                for i in range(len(column_data)):
                    value = column_data[i]
                    if (value == 254 or value == 255):
                        print("index: " + str(i) + "error vaule " + value)
                    elif (math.isnan(value)):
                        print("index: " + str(i) + " NaN value")
                        if (_soc == 100):
                            column_data[i] = 4
                        elif (_speed == 0 and _crt < 0):
                            # 车速为 0， 总电流 <0，为停车充电
                            column_data[i] = 1
                        elif ((_speed > 0 and _Vstatus == 1) or
                              (_speed == 0 and _crt == 0 and _Vstatus == 2)):
                            # 车速 > 0 且 车辆为启动状态， 未充电
                            # 车速 = 0 且 车辆熄火 总电流 = 0， 未充电
                            column_data[i] = 3
                        else:
                            if _speed > 0:
                                # 行驶充电
                                column_data[i] = 2
                            else:
                                if _crt < 0:
                                    # 停车充电
                                    column_data[i] = 1
                                else:
                                    # 未充电
                                    column_data[i] = 3
            elif (c == 'mode'):
                for i in range(len(column_data)):
                    value = column_data[i]
                    if (value != 1 or math.isnan(value)):
                        # 根据提供的数据集描述，所有车辆都应该是纯电动车
                        column_data[i] = 1
            elif (c == 'total_volt'):
                pass
            elif (c == 'total_current'):
                pass
            elif (c == 'mileage'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        value1 = column_data[start - 1]
                        value2 = column_data[end + 1]
                        if end - start == 0:
                            if (not (abnml_mileage(value1))
                                    and not (abnml_mileage(value2))):
                                column_data[start] = (value1 + value2) / 2
                        elif end - start > 0:
                            # 连续空值 (默认mileage一定是递增的)
                            if (not (abnml_mileage(value1))
                                    and not (abnml_mileage(value2))):
                                inter_value = np.linspace(column_data[start -
                                                                      1],
                                                          column_data[end + 1],
                                                          num=end - start + 1)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                            else:
                                pass
                for i in range(len(column_data)):
                    value = column_data[i]
                    if abnml_mileage(value):
                        pass
            elif (c == 'standard_soc'):
                _Vstatus = data['vehicle_state'][i]
                _crt = data['total_current'][i]
                for i in range(len(column_data)):
                    value = column_data[i]
                    if (value == 254 or value == 255 or value < 0
                            or value > 100 or math.isnan(value)):
                        # 如果是异常值或者空值
                        if (_crt == 0 and _Vstatus == 2):
                            # 在电流为 0 车辆熄火的情况下 soc不会有变化
                            if (i != 0 and i != len(column_data)):
                                column_data[i] = column_data[i - 1]
                            elif (i == 0):
                                column_data[i] = column_data[i + 1]
                            elif (i == len(column_data)):
                                column_data[i] = column_data[i - 1]
                            else:
                                column_data[i] = 100
                        elif ((_crt < 0 and _Vstatus == 2)
                              or (_crt > 0 and _Vstatus == 1)):
                            # 车辆停车充电or 车辆启动放电 取前后均值
                            column_data[i] = (column_data[i - 1] +
                                              column_data[i + 1]) / 2
                        else:
                            # 其他情况 取前一个记录的值
                            column_data[i] = column_data[i - 1]
            elif (c == 'max_cell_volt'):
                # 取中位数
                med = np.median(column_data)
                if is_NaN:
                    column_data = pd.DataFrame(column_data)
                    column_data = column_data.fillna(med)
            elif (c == 'min_cell_volt'):
                # 取中位数
                med = np.median(column_data)
                if is_NaN:
                    column_data = pd.DataFrame(column_data)
                    column_data = column_data.fillna(med)
            elif (c == 'max_temp'):
                # 取中位数
                med = np.median(column_data)
                if is_NaN:
                    column_data = pd.DataFrame(column_data)
                    column_data = column_data.fillna(med)
            elif (c == 'min_temp'):
                # 取中位数
                med = np.median(column_data)
                if is_NaN:
                    column_data = pd.DataFrame(column_data)
                    column_data = column_data.fillna(med)
            elif (c == 'gear'):
                pass


if __name__ == '__main__':
    path = '/Users/xiejian/ncbdc/vb_data/0dd8d9a9c15a45beb6db8f7fe8f5ba62/part-00000-66a9d65e-cad2-4f62-af22-e9acbec50dbc.c000.csv'
    # process_sigle(path)
    analysis(path)
    # df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1],
    #                    [np.nan, np.nan, np.nan, 5], [np.nan, 3, np.nan, 4]],
    #                   columns=list('ABCD'))
    # print(df)
    # df = df.fillna(method='pad')
    # df = df.fillna(method='bfill')
    # print(df)
    # df = pd.DataFrame([np.nan, 1, 2, np.nan, np.nan, 3, np.nan, np.nan, np.nan, 4])
    # print(df.index[np.where(np.isnan(df))[0]].values.tolist())
