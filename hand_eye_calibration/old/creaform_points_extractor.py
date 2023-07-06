import pandas as pd
import numpy as np
import csv

file = '/home/lorenzo/Desktop/he_test_CREA.CSV'
data = pd.read_csv(file, usecols=[0,1,2,3,4,5,6,7])
data = data.to_numpy()

value=[]
value.append(np.mean(data[0:42,:], axis=0))
value.append(np.mean(data[72:349,:], axis=0))
value.append(np.mean(data[373:937,:], axis=0))
value.append(np.mean(data[965:1255,:], axis=0))
value.append(np.mean(data[1294:1542,:], axis = 0))
value.append(np.mean(data[1608:1839,:], axis = 0))
value.append(np.mean(data[1861:2084,:], axis = 0))
value.append(np.mean(data[2095:2345,:], axis = 0))
value.append(np.mean(data[2363:2648,:], axis = 0))
value.append(np.mean(data[2672:2938,:], axis = 0))
value.append(np.mean(data[2949:3229, :], axis = 0))
value.append(np.mean(data[3242:3530, :], axis = 0))
value.append(np.mean(data[3576:3914, :], axis = 0))
value.append(np.mean(data[3932:4724, :], axis = 0))
value.append(np.mean(data[4757:4776, :], axis = 0)) #This is the uncertain one.
value.append(np.mean(data[4795:5086, :], axis = 0))
value.append(np.mean(data[5103:5324, :], axis = 0))
value.append(np.mean(data[5340:5548, :], axis = 0))
value.append(np.mean(data[5614:5832, :], axis = 0))
value.append(np.mean(data[5842:6209, :], axis = 0))

for i in range(20):
    print("value N.: ", i," is", value[i])

with open('creaform_collected_points', 'w') as file:
    writer = csv.writer(file)
    for i in range(20):
        print("lel\n",value[i])
        writer.writerow(value[i])