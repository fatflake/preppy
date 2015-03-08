## Input/Output functions for zt_main.py
########################################

import pandas as pd
from matplotlib import pylab as plt
import numpy as np

def read_data():

    spree = np.array(pd.read_csv('../data/spree.csv'))
    gate  = np.array(pd.read_csv('../data/gate.csv'))
    sattelite = np.array(pd.read_csv('../data/sattelite.csv'))
    return spree, gate, sattelite


def plot_everything(spree_x, spree_y, gate_x, gate_y, sattelite_x, sattelite_y):
    plt.figure()
    plt.title("Berlin ROI")    
    plt.plot(spree_x, spree_y)
    plt.plot(spree_x, spree_y, 'bo')           
    plt.plot(gate_x, gate_y, 'go') 
    plt.plot(sattelite_x, sattelite_y, 'yo')   
    plt.plot(sattelite_x, sattelite_y, 'y')    
    plt.savefig('Berlin_ROI.png')
