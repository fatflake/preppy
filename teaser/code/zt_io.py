## Input/Output functions for zt_main.py
########################################

import pandas as pd
from matplotlib import pylab as plt
import numpy as np

X_MIN, X_MAX = 0., 20.
Y_MIN, Y_MAX = -5., 15.

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


def plot_joint(probs, location):    
    print "location", location[0], location[1], probs[location[0], location[1]]
    fig,ax = plt.subplots() 
    plt.title("Map of probability of candidate")  
    probs = probs[:,::-1]  
    print "probs.min(): ", probs.min()
    print "probs.max(): ", probs.max()
    im = ax.imshow(probs.T, vmin=probs.min(), vmax=probs.max(), #vmin=abs(probs).min(), vmax=abs(probs).max(), 
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], interpolation="None") #cmap="Greys",
    probs[location[0], location[1]] = 0 # FIXME
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax) 
    #plt.scatter(location[0], location[1], 'yo')
    #plt.show()
    plt.savefig('probability_map_joint.png')   

def plot_probs(probs):
    fig,ax = plt.subplots() 
    plt.title("Map of probability element")  
    probs = probs[:,::-1]  
    print "probs.min(): ", probs.min()
    print "probs.max(): ", probs.max()
    im = ax.imshow(probs.T, vmin=probs.min(), vmax=probs.max(), #vmin=abs(probs).min(), vmax=abs(probs).max(), 
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], interpolation="None") #cmap="Greys",
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax) 

def plot_individual(spree, gate, satt):
    plot_probs(spree)
    plt.savefig('probability_map_spree.png')   
    plot_probs(gate)
    plt.savefig('probability_map_gate.png')   
    plot_probs(satt)
    plt.savefig('probability_map_satt.png')   

