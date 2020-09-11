import pandas as pd
import math


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
    _speed = data['speed']
    _Vstatus = data['vehicle_state']
    _Cstatus = data['charging_status']
    _Tcurrent = data['total_current']
    _Tvolt = data['total_volt']
    _mile = data['mileage']
    _soc = data['standard_soc']
    for s, vtt, ctt, c, v, m, soc in zip(_speed, _Vstatus, _Cstatus, _Tcurrent,
                                         _Tvolt, _mile, _soc):
        # vehicle_state
        # if (s == 0.0 and ctt == 3 and c <= 0):
        #    print('Vstatus= ' + str(vtt) + ' _Tcurrent= ' + str(c) +
        #          ' _Tvolt = ' + str(v))
        # else:
        #     pass
        if (ctt == 3 and vtt == 2):
            print('speed= ' + str(s) + ' Vstatus= ' + str(vtt) +
                  ' _Tcurrent= ' + str(c))


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
        # is_NaN = column_data.isna().any()
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
            elif (c == "charging_status"):
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


if __name__ == '__main__':
    path = '/Users/xiejian/ncbdc/vb_data/0dd8d9a9c15a45beb6db8f7fe8f5ba62/part-00000-66a9d65e-cad2-4f62-af22-e9acbec50dbc.c000.csv'
    # process_sigle(path)
    analysis(path)
