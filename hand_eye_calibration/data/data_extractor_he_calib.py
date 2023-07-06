import numpy as np
import glob, os, sys
import pandas as pd
import pathlib
import natsort

def read_crea_data(filename):
    df = pd.read_csv(filename, skiprows=1, usecols=[1,2,3,7], header=0, names=['TX', 'TY', 'TZ', 'Valid'])
    data = pd.DataFrame(df)
    data = data.drop(data[data.Valid == 0].index)
    data = data[['TX', 'TY', 'TZ']].to_numpy()

    #Remember to divide these points by 100 because hey are in millimeters, not in meters.
    data = data/1000.

    return data

def extract_creaform_points():
    #Extract point from Creaform
    points = []
    for dir in natsort.natsorted((pathlib.Path(__file__).parent / "creaform").glob("*.CSV")):
        print("Extracting creaform", dir)
        points.append(np.mean(read_crea_data(dir), 0))

    return np.asarray(points)
    
def read_UR_data(filename):
    #TX, TY, TZ for the robot
    data = pd.DataFrame(pd.read_csv(filename))
    data = data[["parent_frame", "child_frame", "posx", "posy", "posz"]]
    data = data[["posx", "posy", "posz"]].to_numpy()

    return data

def extract_UR_data():
    #Extract point from UR
    points = []
    for gen_dir in natsort.natsorted((pathlib.Path(__file__).parent / "UR").iterdir()):
        if gen_dir.is_dir():
            print("Opening UR dir ", gen_dir)
            for dir in natsort.natsorted(gen_dir.glob("*.csv")):
                print("Analyzing \n", dir)
                points.append(np.mean(read_UR_data(dir), 0))

    return np.asarray(points)

creaform_points = extract_creaform_points()
UR_points = extract_UR_data()

df = pd.DataFrame(np.hstack((creaform_points,UR_points)), columns=['cx', 'cy', 'cz', 'urx', 'ury', 'urz'])

df.to_csv(pathlib.Path(__file__).parent.parent / "he_calibration_data.csv", float_format='%.3f')





