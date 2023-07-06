import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def load_file(filename):
    #loading data from CSV
    data = pd.read_csv(filename, skiprows = 1, converters={'TY': lambda x: str(x), 'TZ': lambda x: str(x)})
    df = pd.DataFrame(data)
    ts = data.index; ts = ts.to_numpy(); ts = ts.reshape(-1,1)

    vals = df.to_numpy()

    for i, value in enumerate(vals[:]):
        if value[0] == 0 and value[1] == 0:
            np.delete(vals, (i), axis = 0)

    for i in vals[:]:
        a = i[2]; b = i[3]
        i[2] = np.float64(a+b)
        
    vals = np.delete(vals, 3, 1)
    vals = np.hstack((ts, vals))
    
    return vals

def time_plot(file_directory):
    all_data = load_file(file_directory)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle('End Effector Position')
    
    ax = fig.add_subplot(3,2,1)
    ax.plot(all_data[:,0], all_data[:,1], color="blue")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("x [mm]")
    ax.grid(visible=True)

    ax = fig.add_subplot(3,2,3)
    ax.plot(all_data[:,0], all_data[:,3], color="green")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("y [mm]")
    ax.grid(visible=True)

    ax = fig.add_subplot(3,2,5)
    ax.plot(all_data[:,0], -all_data[:,2], color="red")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [mm]")
    ax.grid(visible=True)

    ax = fig.add_subplot(1,2,2, projection="3d")
    ax.scatter(all_data[:,1],all_data[:,3],-all_data[:,2])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    time_plot("/home/lorenzo/Desktop/Ringsted Trials/test01_T2100.CSV")