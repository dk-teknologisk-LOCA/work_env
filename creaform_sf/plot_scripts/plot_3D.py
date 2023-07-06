import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import glob
import os

def load_file(filename):
    #loading data from CSV
    data = pd.read_csv(filename, skiprows = 1, converters={'TY': lambda x: str(x), 'TZ': lambda x: str(x)})
    df = pd.DataFrame(data)
    vals = df.to_numpy() #from the first column onwards because    

    for i, value in enumerate(vals[:]):
        if value[0] == 0 and value[1] == 0:
            np.delete(vals, (i), axis = 0)

    for i in vals[:]:
        a = i[2]; b = i[3]
        i[2] = np.float64(a+b)
        
    vals = np.delete(vals, 3, 1)
    print("tipo", type(vals))
    return vals

def load_data(foldername):
    data = []
    file_list = glob.glob(os.path.join(foldername,"*.CSV"))

    for file in file_list:
        vals = load_file(file)
        data.append(vals)

    vals = load_file(file_list[0])
    data.append(vals)
    
    return data

def DDD_plot(file_directory):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # get data #
    all_data = load_data(file_directory)

    for data in all_data:
        # plot data #
        xdata = data[:,0]
        ydata = data[:,1]
        zdata = data[:,2]
        ax.scatter(xdata, ydata, zdata)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

if __name__ == "__main__":
    DDD_plot("/home/lorenzo/Desktop/Sequences with tool/")




def DDD_plot_example():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    

    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')

    # Data for three-dimensional scattered points
    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()