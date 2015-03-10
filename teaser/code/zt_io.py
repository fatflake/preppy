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
    plt.plot(spree[0], spree[1], 'bo')           
    plt.plot(gate[0], gate[1], 'go') 
    plt.plot(sattelite[0], sattelite[1], 'yo')   
    plt.plot(sattelite[0], sattelite[1], 'y')    
    plt.savefig('Berlin_ROI.png')


def plot_result(probs, location):
    X_MIN, X_MAX = 0., 20.
    Y_MIN, Y_MAX = -5., 15.
    #plt.figure()
    
    print "location", location[0], location[1], probs[location[0], location[1]]
    probs[location[0], location[1]] = 0
    fig,ax = plt.subplots() 
    plt.title("Map of probability of candidate")  
    probs = probs[:,::-1]  
    im = ax.imshow(probs.T, cmap="Greys", vmin=abs(probs).min(), vmax=abs(probs).max(), 
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], interpolation="None")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax) 
    #plt.scatter(location[0], location[1], 'yo')
    #plt.show()
    plt.savefig('probability_map.png')   


