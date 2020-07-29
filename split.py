import pandas as pd
import numpy as np
import time
import torch

# timeStepH = []
# timeStepNH = []

h1 = []
nh1 = []
h2 = []
nh2 = []
h3 = []
nh3 = []
h4 = []
nh4 = []
h5 = []
nh5 = []
h6 = []
nh6 = []

def time_step(x,y,p):
	while x <= y:
		try:
			path = p + str(x) + '.csv'
			data = pd.read_csv(path, header=None)
			if len(data) > 0: # exist at least one record
				time_data = data.iloc[0,2]
				timeArray = time.strptime(time_data, "%Y-%m-%d %H:%M:%S")
				time_data = time.strftime("%Y/%m/%d", timeArray)
				data = data.iloc[:, 2:18]
				data = data.values.tolist()
				if (time_data == '2019/09/11' or time_data == '2019/09/12'):
					if p == './h/h':
						h1.append(data)
					else:
						nh1.append(data)
				elif (time_data == '2019/10/10' or time_data == '2019/10/11' or time_data =='2019/10/12'):
					if p == './h/h':
						h2.append(data)
					else:
						nh2.append(data)
				elif (time_data == '2018/11/10' or time_data == '2018/11/11' or time_data == '2018/11/12'):
					if p == './h/h':
						h3.append(data)
					else:
						nh3.append(data)
				elif (time_data == '2019/01/11' or time_data == '2019/01/12'):
					if p == './h/h':
						h4.append(data)
					else:
						nh4.append(data)
				elif (time_data == '2018/12/10' or time_data == '2018/12/11' or time_data == '2018/12/12'):
					if p == './h/h':
						h5.append(data)
					else:
						nh5.append(data)
				elif (time_data == '2019/08/10' or time_data == '2019/08/11' or time_data == '2019/08/12'):
					if p == './h/h':
						h6.append(data)
					else:
						nh6.append(data)
				else:
					pass
		except:
			pass
		x += 1

def to_csv(h,nh,time):
	for i in range(len(h)):
		_csv = pd.DataFrame(data = np.array(h[i]))
		csv_path_h = 'D:/shortTerm/' + time + '/h/h' + str(i + 1) + '.csv'
		_csv.to_csv(csv_path_h, index = False, header = False)

	for i in range(len(nh)):
		_csv = pd.DataFrame(data = np.array(nh[i]))
		csv_path_nh = 'D:/shortTerm/' + time + '/nh/nh' + str(i + 1) + '.csv'
		_csv.to_csv(csv_path_nh, index = False, header = False)

if __name__ == '__main__':
	h_path = './h/h'
	nh_path = './nh/nh'

	time_step(2,237,h_path)
	time_step(2,520,nh_path)

	# (2) 2019-10
	to_csv(h6,nh6,'2019-08')