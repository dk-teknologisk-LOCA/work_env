import matplotlib.pyplot as plt
import numpy as np

f = open('25_08_2022_13_01_54.txt', 'r')
f_lines = f.readlines()
time = []
x = []
y = []
z = []
rx = []
ry = []
rz = []
rw = []
data = []

i = 0
for line in f_lines:
    l_split = line.split(';')
    time.append(int(l_split[0]))
    x.append(float(l_split[1]))
    data.append([float(a) for a in l_split])
    # print(line)

st = time[0]

for i in range(0, len(time)):
    time[i] = time[i]-st
    data[i][0] = data[i][0] -st

print(np.array(data)[:,0])
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(np.array(data)[:,0], np.array(data)[:,1])
ax2.plot(np.array(data)[:,0], np.array(data)[:,2])
ax3.plot(np.array(data)[:,0], np.array(data)[:,3])
plt.show()
