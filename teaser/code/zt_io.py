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


def plot_everything(spree, gate, sattelite):
    plt.figure()
    plt.title("Berlin ROI")    
    plt.plot(spree[0], spree[1])
    plt.plot(spree[0], spree[1], 'ro')           
    plt.plot(gate[0], gate[1], 'go') 
    plt.plot(sattelite[0], sattelite[1], 'yo')   
    plt.plot(sattelite[0], sattelite[1], 'y')    
    plt.savefig('Berlin_ROI.png')
