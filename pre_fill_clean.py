import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


cdtype = {
    'yr_modahrmn': str,
    'message_type': str,
    # 'speed': float,
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
    'bat_fault_list': object,
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
    'alarm_info': str
}


def check_nan(path):
    try:
        data = pd.read_csv(
            path,
            header=0,
            dtype=cdtype)
    except Exception as e:
        print(str(e))
    else:
        for c in data.columns:
            column_data = data[c]
            if column_data.isna().any():
                if column_data.dropna().empty:
                    print(c + ' ->IS EMPTY')
                else:
                    print(c + ' ->GET NaN VALUE')
            else:
                print(c + ' ->FILL')


def analysis(path):
    try:
        data = pd.read_csv(
            path,
            header=0,
            dtype=cdtype)
    except Exception as e:
        print(str(e))
    _speed = np.array(data['speed'].values.tolist())[100:5000]
    _speed = _speed / 100
    # _Vstatus = np.array(data['vehicle_state'].values.tolist())
    _Cstatus = np.array(data['charging_status'].values.tolist())
    _Tcurrent = np.array(data['total_current'].values.tolist())[100:5000]
    _Tcurrent = _Tcurrent / 100
    _Tvolt = np.array(data['total_volt'].values.tolist())
    _Tvolt = _Tvolt / 100
    # _mile = np.array(data['mileage'].values.tolist())
    # _soc = np.array(data['standard_soc'].values.tolist())
    # _soc = _soc / 100
    # _maxvolt = np.array(data['max_cell_volt'].values.tolist())
    # _minvolt = np.array(data['min_cell_volt'].values.tolist())
    # _maxtemp = np.array(data['max_temp'].values.tolist())
    # _mintemp = np.array(data['min_temp'].values.tolist())
    # _gear = np.array(data['gear'].values.tolist())
    # index = np.arange(0, len(data), 1)
    index = np.arange(100, 5000, 1)
    plt.plot(index, _speed, label='speed')
    # plt.plot(index, _Vstatus, label='vehicle stt')
    # plt.plot(index, _Cstatus, label='charge stt')
    # plt.plot(index, _soc, label='soc')
    # plt.plot(index, _Tvolt, label='volt')
    plt.plot(index, _Tcurrent, label='current')
    # plt.plot(index, _mile, label='mile')
    # plt.plot(index, _maxvolt, label='max_volt')
    # plt.plot(index, _minvolt, label='min_volt')
    # plt.plot(index, _maxtemp, label='max_temp')
    # plt.plot(index, _mintemp, label='min_temp')
    # plt.plot(index, _gear, label='gear')
    plt.legend()
    plt.grid()
    plt.show()


def process_sigle(path):
    print(path)
    try:
        data = pd.read_csv(
            path,
            header=0,
            dtype=cdtype)
    except Exception as e:
        print(str(e))
    for c in data.columns:
        # 获取列（数据项）及缺失值情况
        column_data = data[c]
        # 不处理这些数据项
        if (c == 'cell_volt_list' or c == 'bat_fault_list'
                or c == 'cell_temp_list' or c == 'alarm_info'):
            continue
        # 是否存在缺失值
        is_NaN = column_data.isna().any()
        if is_NaN:
            print('Processing: ' + c)
            # 获取所有缺失值索引
            nan_list = column_data.index[np.where(
                np.isnan(column_data))[0]].values.tolist()
            # co_list记录连续缺失值的开始结束位置
            co_list = []
            # fNaN_list由多个co_list组成，用于定位列表中所有缺失值
            fNaN_list = []
            for d in nan_list:
                co_list.append(d)
                if d + 1 not in nan_list:
                    fNaN_list.append([co_list[0], co_list[-1]])
                    co_list = []
        # 是否整个数据项为空
        is_empty = column_data.dropna().empty
        column_data = column_data.values.tolist()
        if is_empty:
            # 处理整个数据项为空的情况
            pass
        else:
            if (c == 'yr_modahrmn' or c == 'message_type'):
                pass
            elif (c == 'speed'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                            # 参考vehicle_state(车辆状态)和charging_status(充电状态)在缺失值索引处对应的值
                            _Vstate = data['vehicle_state'][start]
                            _Cstatus = data['charging_status'][start]
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                # 单一缺失值
                                if (_Vstate - 1 == 0 or _Cstatus - 2 == 0
                                        or _Cstatus - 4 == 0):
                                    column_data[start] == 0
                                else:
                                    column_data[start] == (b_value +
                                                           f_value) / 2
                            else:
                                # 连续缺失值
                                # 缺失个数
                                n = end - start + 1
                                # 生成等差填补序列
                                inter_value = np.linspace(b_value,
                                                          f_value,
                                                          num=n)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                else:
                    pass
                # 替换
                data['speed'] = pd.DataFrame(column_data)
            elif (c == 'vehicle_state'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        for d in range(start, end + 1):
                            try:
                                # 或取参照数据（其他数据项，缺失值前后的有效记录）
                                b_value = column_data[start - 1]
                                f_value = column_data[end + 1]
                                _speed = data['speed'][d]
                                _Cstatus = data['charging_status'][d]
                                _crt = data['total_current'][d]
                            except Exception as e:
                                print('文件: ' + path + ' 无法获取参照数据,具体错误' +
                                      str(e))
                            else:
                                if ((_speed == 0 and _Cstatus == 1)
                                        or (_speed == 0 and _Cstatus == 4)
                                        or (_speed == 0 and _Cstatus == 3
                                            and _crt <= 0)):
                                    column_data[d] = 2
                                elif (_speed > 0 and _Cstatus == 3
                                      and _crt > 0):
                                    column_data[d] = 1
                                else:
                                    column_data[d] = column_data[d - 1]
                else:
                    pass
                # 替换
                data['vehicle_state'] = pd.DataFrame(column_data)
            elif (c == 'charging_status'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        for d in range(start, end + 1):
                            try:
                                # 或取参照数据（其他数据项，缺失值前后的有效记录）
                                b_value = column_data[start - 1]
                                f_value = column_data[end + 1]
                                _speed = data['speed'][d]
                                _Vstate = data['vehicle_state'][d]
                                _crt = data['total_current'][d]
                            except Exception as e:
                                print('文件: ' + path + ' 无法获取参照数据,具体错误' +
                                      str(e))
                            else:
                                if (_speed == 0 and _Vstate == 2 and _crt < 0):
                                    column_data[d] = 1
                                elif (_speed == 0 and _Vstate == 2
                                      and _crt == 0):
                                    column_data[d] = 4
                                elif (_speed > 0 and _Vstate == 1):
                                    column_data[d] = 3
                                else:
                                    column_data[d] = column_data[d - 1]
                else:
                    pass
                # 替换
                data['charging_status'] = pd.DataFrame(column_data)
            elif (c == 'mode'):
                if is_NaN:
                    # 根据提供的数据集描述，所有车辆都应该是纯电动车
                    column_data = pd.DataFrame(column_data).fillna(1)
                    column_data = column_data.values.tolist()
                else:
                    pass
                # 替换
                data['mode'] = pd.DataFrame(column_data)
            elif (c == 'total_volt'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = (b_value + f_value) / 2
                            else:
                                # 连续缺失值
                                # 缺失个数
                                n = end - start + 1
                                # 生成等差填补序列
                                inter_value = np.linspace(b_value,
                                                          f_value,
                                                          num=n)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                else:
                    pass
                # 替换
                data['total_volt'] = pd.DataFrame(column_data)
            elif (c == 'total_current'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                            _Ctt_list = data['charging_status'][
                                start:end].values.tolist()
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = (b_value + f_value) / 2
                            else:
                                if (np.mean(_Ctt_list) == 1
                                        or np.mean(_Ctt_list) == 4
                                        or (max(_Ctt_list) == 4)
                                        and min(_Ctt_list) == 1):
                                    # 连续缺失值
                                    # 缺失个数
                                    n = end - start + 1
                                    # 生成等差填补序列
                                    inter_value = np.linspace(b_value,
                                                              f_value,
                                                              num=n)
                                    p = 0
                                    for d in range(start, end + 1):
                                        column_data[d] = inter_value[p]
                                        p = p + 1
                                else:
                                    for d in range(start, end + 1):
                                        column_data[d] = (b_value +
                                                          f_value) / 2
                else:
                    pass
                # 替换
                data['total_current'] = pd.DataFrame(column_data)
            elif (c == 'mileage'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        # 或取参照数据（其他数据项，缺失值前后的有效记录）
                        b_value = column_data[start - 1]
                        f_value = column_data[end + 1]
                        if (end - start == 0):
                            column_data[start] = (b_value + f_value) / 2
                        else:
                            # 连续缺失值
                            # 缺失个数
                            n = end - start + 1
                            # 生成等差填补序列
                            inter_value = np.linspace(b_value, f_value, num=n)
                            p = 0
                            for d in range(start, end + 1):
                                column_data[d] = inter_value[p]
                                p = p + 1
                else:
                    pass
                # 替换
                data['mileage'] = pd.DataFrame(column_data)
            elif (c == 'standard_soc'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                            _Ctt_list = data['charging_status'][
                                start:end].values.tolist()
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = int(
                                    (b_value + f_value) / 2)
                            else:
                                if (np.mean(_Ctt_list) == 1
                                        or min(_Ctt_list) > 1):
                                    # 缺失个数
                                    n = end - start + 1
                                    # 生成等差填补序列
                                    inter_value = np.linspace(b_value,
                                                              f_value,
                                                              num=n)
                                    p = 0
                                    for d in range(start, end + 1):
                                        column_data[d] = inter_value[p]
                                        p = p + 1
                                elif (_Ctt_list[0] - _Ctt_list[-1] == 2
                                      and max(_Ctt_list) == 3
                                      and min(_Ctt_list) == 1):
                                    # 缺失个数
                                    n = end - start + 1
                                    n1 = 0
                                    for x in _Ctt_list:
                                        if x != 1:
                                            n1 = n1 + 1
                                    n2 = n - n1
                                    inter_value_1 = np.linspace(b_value,
                                                                50,
                                                                num=n1)
                                    inter_value_2 = np.linspace(50,
                                                                f_value,
                                                                num=n2)
                                    inter_value = np.append(
                                        inter_value_1, inter_value_2)
                                    p = 0
                                    for d in range(start, end + 1):
                                        column_data[d] = inter_value[p]
                                        p = p + 1
                                else:
                                    column_data[d] = column_data[d - 1]
                else:
                    pass
                # 替换
                data['standard_soc'] = pd.DataFrame(column_data)
            elif (c == 'max_cell_volt'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = (b_value + f_value) / 2
                            else:
                                # 连续缺失值
                                # 缺失个数
                                n = end - start + 1
                                # 生成等差填补序列
                                inter_value = np.linspace(b_value,
                                                          f_value,
                                                          num=n)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                else:
                    pass
                # 替换
                data['max_cell_volt'] = pd.DataFrame(column_data)
            elif (c == 'min_cell_volt'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = (b_value + f_value) / 2
                            else:
                                # 连续缺失值
                                # 缺失个数
                                n = end - start + 1
                                # 生成等差填补序列
                                inter_value = np.linspace(b_value,
                                                          f_value,
                                                          num=n)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                else:
                    pass
                # 替换
                data['min_cell_volt'] = pd.DataFrame(column_data)
            elif (c == 'max_temp'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = (b_value + f_value) / 2
                            else:
                                # 连续缺失值
                                # 缺失个数
                                n = end - start + 1
                                # 生成等差填补序列
                                inter_value = np.linspace(b_value,
                                                          f_value,
                                                          num=n)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                else:
                    pass
                # 替换
                data['max_temp'] = pd.DataFrame(column_data)
            elif (c == 'min_temp'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        try:
                            # 或取参照数据（其他数据项，缺失值前后的有效记录）
                            b_value = column_data[start - 1]
                            f_value = column_data[end + 1]
                        except Exception as e:
                            print('文件: ' + path + ' 无法获取参照数据,具体错误' + str(e))
                        else:
                            if (end - start == 0):
                                column_data[start] = (b_value + f_value) / 2
                            else:
                                # 连续缺失值
                                # 缺失个数
                                n = end - start + 1
                                # 生成等差填补序列
                                inter_value = np.linspace(b_value,
                                                          f_value,
                                                          num=n)
                                p = 0
                                for d in range(start, end + 1):
                                    column_data[d] = inter_value[p]
                                    p = p + 1
                else:
                    pass
                # 替换
                data['min_temp'] = pd.DataFrame(column_data)
            elif (c == 'gear'):
                if is_NaN:
                    for r in fNaN_list:
                        start = r[0]
                        end = r[-1]
                        for d in range(start, end + 1):
                            try:
                                # 或取参照数据（其他数据项，缺失值前后的有效记录）
                                b_value = column_data[start - 1]
                                f_value = column_data[end + 1]
                                _speed = data['speed'][d]
                                _Vstate = data['vehicle_state'][d]
                            except Exception as e:
                                print('文件: ' + path + ' 无法获取参照数据,具体错误' +
                                      str(e))
                            else:
                                if (_speed == 0 and _Vstate == 2):
                                    column_data[d] = 0
                                elif (_speed == 0 and _Vstate == 1):
                                    column_data[d] = 15
                                else:
                                    column_data[d] = 14
                else:
                    pass
                # 替换
                data['gear'] = pd.DataFrame(column_data)
            elif (c == 'max_volt_num'):
                if is_NaN:
                    # 用众数填充
                    _mode = data['max_volt_num'].mode()
                    column_data = pd.DataFrame(column_data).fillna(_mode)
                    column_data = column_data.values.tolist()
                else:
                    pass
                # 替换
                data['max_volt_num'] = pd.DataFrame(column_data)
            elif (c == 'min_volt_num'):
                if is_NaN:
                    # 用众数填充
                    _mode = data['min_volt_num'].mode()
                    column_data = pd.DataFrame(column_data).fillna(_mode)
                    column_data = column_data.values.tolist()
                else:
                    pass
                # 替换
                data['min_volt_num'] = pd.DataFrame(column_data)
            elif (c == 'max_temp_num'):
                if is_NaN:
                    # 用众数填充
                    _mode = data['max_temp_num'].mode()
                    column_data = pd.DataFrame(column_data).fillna(_mode)
                    column_data = column_data.values.tolist()
                else:
                    pass
                # 替换
                data['max_temp_num'] = pd.DataFrame(column_data)
            elif (c == 'min_temp_num'):
                if is_NaN:
                    # 用众数填充
                    _mode = data['min_temp_num'].mode()
                    column_data = pd.DataFrame(column_data).fillna(_mode)
                    column_data = column_data.values.tolist()
                else:
                    pass
                # 替换
                data['min_temp_num'] = pd.DataFrame(column_data)
            else:
                pass
    return data


if __name__ == '__main__':
    path = '/Users/xiejian/ncbdc/vb_data/0dd8d9a9c15a45beb6db8f7fe8f5ba62/part-00000-66a9d65e-cad2-4f62-af22-e9acbec50dbc.c000.csv'
    # fill_data = process_sigle(path)
    # # analysis(path)
    # fill_data.to_csv('/Users/xiejian/ncbdc/test_fill.csv')
    # check_nan(path)
    # print('\naftet fill\n')
    # check_nan('/Users/xiejian/ncbdc/test_fill.csv')
    analysis(path)
