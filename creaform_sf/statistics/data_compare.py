#Code to compare the reference data from the robot movement with the data
# from the average of the observation of the test runs

#Suppose we have the actual movement from LH in the same format as the other file, so
#x y z Roll Pitch Yaw.
  
from cgi import test
import numpy as np
import scipy.stats
import glob 
import os
from matplotlib.pyplot import plot, figure, xlabel, ylabel, legend, show, grid, subplot, annotate
import pandas as pd

col_lbls = {0: "X", 1 : "Y", 2 : "Z", 
            3 : "Yaw", 4 : "Pitch", 5 : "Roll"}

def load_data(foldername):
    #loading data from Excel
    data = pd.read_excel(foldername)
    df = pd.DataFrame(data)
    vals = df.to_numpy()
    print("vals shape", vals.shape)
    return vals

def plottante(test_data, UR_data, qt, t):
    #dim is which column we want to plot, voci is a dictionary 
    # of the column labels of the matrix
    
    if qt == 2:
        figure()
        for i in range(1,7):
            n = 230+i
            subplot(n)
            plot(t, test_data[:, i-1], label="test_"+col_lbls[i-1])
            plot(t, UR_data[:, i-1], label="UR_"+col_lbls[i-1])
            legend()
            grid()
        show()
    elif qt == 1:
        figure()
        for i in range(1,7):
            n = 230+i
            subplot(n)
            plot(t, test_data[:, i-1], label="offset_"+col_lbls[i-1])
            annotate("Max: {:.4}".format(np.max(test_data[:,i-1])), (np.argmax(test_data[:,i-1]), np.max(test_data[:,i-1])))
            legend()
            grid()
        show()

def check_precision(test_data, UR_data):
    diff = test_data-UR_data
    max_diff = np.max(diff)

    return diff, max_diff

if __name__ == "__main__":
    foldername = "folder with the result of the tests"
    foldername_og = "folder with UR data"

    test_data = load_data("/home/lorenzo/data_testing/avg_run.ods")
    UR_data = load_data("/home/lorenzo/data_testing/source/viper_data_two.ods")
    t = np.arange(0, test_data.shape[0])
    #Data should be in a N samples (rows) by 6 dimensions (columns)
    # matrix, if not, convert it
    #test_data = np.random.rand(5000, 6)
    #UR_data = np.random.rand(5000, 6)

    plottante(test_data, UR_data, 2, t)

    offset, max_offset = check_precision(test_data, UR_data)

    print("Max Offset", max_offset)
    plottante(offset, None, 1, t)




