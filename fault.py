import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
import math

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


def temp_diff(path):
    try:
        data = pd.read_csv(path, header=0, dtype=cdtype)
        alarm_LVL = data['max_alarm_lvl']
        alarm_INFO = data['alarm_info']
        max_TEMP = data['max_temp']
        min_TEMP = data['min_temp']
    except Exception as e:
        print(str(e))
    else:
        diff_list = []
        for alaml, almi, maxT, minT in zip(alarm_LVL, alarm_INFO, max_TEMP,
                                           min_TEMP):
            try:
                temp_diff = maxT - minT
            except Exception as e:
                print('IN: ' + path + " " + str(e))
            else:
                # print('max temp:' + str(maxT) + ' min_temp: ' + str(minT) +
                #       ' diff: ' + str(temp_diff) + ' lvl: ' + str(alaml))
                if (almi == "温度差异报警" and alaml > 0):
                    if (math.isnan(maxT) or math.isnan(minT) or maxT > 210
                            or minT > 210 or maxT < -40 or minT < -40):
                        pass
                    else:
                        diff_list.append(temp_diff)
        return diff_list


def temp(path):
    try:
        data = pd.read_csv(path, header=0, dtype=cdtype)
        TEMP = data['min_temp']
    except Exception as e:
        print('IN: ' + path + " " + str(e))
    else:
        temp_list = []
        for temp in TEMP:
            if math.isnan(temp):
                pass
            else:
                temp_list.append(temp)
        return temp_list


def volt_diff(path):
    try:
        data = pd.read_csv(path, header=0, dtype=cdtype)
        MAX_CELL_V = data['max_cell_volt']
        MIN_CELL_V = data['min_cell_volt']
    except Exception as e:
        print(str(e))
    else:
        diff_list = []
        for maxV, minV in zip(MAX_CELL_V, MIN_CELL_V):
            try:
                diff = maxV - minV
            except Exception as e:
                print('IN: ' + path + " " + str(e))
            else:
                diff_list.append(diff)
        return diff_list


def volt(path, M):
    try:
        data = pd.read_csv(path, header=0, dtype=cdtype)
        if M == 0:
            _VOLT = data['max_cell_volt']
        elif M == 1:
            _VOLT = data['min_cell_volt']
    except Exception as e:
        print('IN: ' + path + " " + str(e))
    else:
        volt_list = []
        for v in _VOLT:
            if math.isnan(v):
                pass
            else:
                volt_list.append(v)
        return volt_list


if __name__ == '__main__':
    root_path = '/home/arron/dataset/battery_04/vb_data/'
    dirs = os.listdir(root_path)
    # dirs.remove('.DS_Store')
    CSV_PATH = []

    for d in dirs:
        file = os.listdir(root_path + d)
        CSV_PATH.append(root_path + d + '/' + file[0])

    # DIFF = []
    # for p in CSV_PATH:
    #     print(p)
    #     DIFF.extend(volt_diff(p))
    # sns.distplot(DIFF)
    # plt.savefig("./volt_diff.png")

    # TEMP = []
    # for p in CSV_PATH:
    #     print(p)
    #     TEMP.extend(temp(p))
    # sns.distplot(TEMP)
    # plt.savefig("./min_temp.png")

    VOLT_0 = []
    for p in CSV_PATH:
        print(p)
        VOLT_0.extend(volt(p, 0))
    sns.distplot(VOLT_0, label='max_cell_volt')
    VOLT_1 = []
    for p in CSV_PATH:
        print(p)
        VOLT_1.extend(volt(p, 1))
    sns.distplot(VOLT_1, label='min_cell_volt')
    plt.savefig("./cell_volt.png")
