import numpy as np
from scipy.optimize import curve_fit as cf
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_dire):
    df = pd.read_csv(file_dire)
    df = df[['SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'FreeAcc_E', 'FreeAcc_N','FreeAcc_U', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Roll', 'Pitch', 'Yaw']]
    df = df.iloc[200:-2000]
    df['SampleTimeFine'] = (df['SampleTimeFine'].iloc[:]-df['SampleTimeFine'].iloc[0])*10**(-4)
    print("df\n", df)

    return df

def f(t, C1, C2, T):
    return C1*(1-np.exp(-np.divide(t,T))) + C2

def rnd_data():
    y= f(df[:,0], 5, 1, 0.5)
    rng = np.random.default_rng()
    y_noise = 0.2 * rng.normal(size=df[:,0].size)
    ydata = y + y_noise
    plt.plot(df[:,0], ydata, 'b-', label='data')

    return ydata

def refitter(t, data):
    params, cov = cf(f, t, data, method='lm')
    curve_refit = f(t, *params)

    return curve_refit, params, cov

def plotter(df, labels, t):
    data = df
    i = 1
    param_list = []
    fig, ax = plt.subplots(3,4)
    for ii in range(3):
        for jj in range(4): 
            refit, params, cov = refitter(t, df[:,i])
            param_list.append(params)
            ax[ii,jj].plot(t, data[:, i], 'b-')
            ax[ii,jj].plot(t, refit, 'g--')
            ax[ii,jj].legend([labels[i], 'fit: C1=%5.3f, C2=%5.3f, T=%5.3f' % tuple(params)])
            i = i + 1

    return fig, param_list

def root_mse(actual, predicted):
    MSE = np.mean(np.square(np.subtract(actual,predicted))) 
    
    RMSE = np.sqrt(MSE)

    return RMSE

def shift_data(data, params, time):
    #note that the first column of the data is the time stamps
    params = np.transpose(params)
    for i in range(data.shape[1]):
        curve = f(time, params[0,i], params[1,i], params[2,i])
        data[:, i] = data[:, i] - curve

    return data

if __name__ == "__main__":
    data = load_data("/home/lorenzo/Documents/biasdata.csv")
    df = data.to_numpy()
    t = df[:,0]

    #Parameter list contains a vector of Nx3 values with N = values fitted (accx, accy, accz, r, p, y, etc, etc) and 3 C1, C2 and T in this order
    fig, parameter_list = plotter(df, data.columns.tolist(), t)
    shifted_data = shift_data(df[:, 1:], parameter_list, t)

    rmses = []
    stds = []
    for i in range(shifted_data.shape[1]):
        rmses.append(root_mse(shifted_data[:,i]), )#predicted e la media o che?
        stds.append(np.std(shifted_data[:,i]))

    print("RMSEs ARE:\n", rmses)
    print("STDs are:\n", stds)
    
    plt.show()




    