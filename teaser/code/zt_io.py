## Input/Output functions for zt_main.py
########################################

import pandas as pd
from matplotlib import pylab as plt
import numpy as np

import pygmaps 

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
    im = ax.imshow(probs.T, vmin=probs.min(), vmax=probs.max(),
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], interpolation="None")# cmap="Greys")
    probs[location[0], location[1]] = 0 # FIXME
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax) 
    plt.savefig('probability_map_joint.png')   

def plot_probs(probs):
    fig,ax = plt.subplots() 
    plt.title("Map of probability element")  
    probs = probs[:,::-1]  
    im = ax.imshow(probs.T, vmin=probs.min(), vmax=probs.max(), 
               extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], interpolation="None")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax) 

def plot_individual(spree, gate, satt):
    plot_probs(spree)
    plt.savefig('probability_map_spree.png')   
    plot_probs(gate)
    plt.savefig('probability_map_gate.png')   
    plot_probs(satt)
    plt.savefig('probability_map_satt.png')   


def plot_gmap(max_pt, gate_arr, spree, satt):
    spree_path = [(spree[i,0], spree[i,1]) for i in range(len(spree)) ]
    satt_path = [(satt[i,0], satt[i,1]) for i in range(len(satt)) ]
    gate = gate_arr[0]
    ########## CONSTRUCTOR: pygmaps.maps(latitude, longitude, zoom) ##############################
    # DESC:         initialize a map  with latitude and longitude of center point  
    #               and map zoom level "15"
    # PARAMETER1:   latitude (float) latittude of map center point
    # PARAMETER2:   longitude (float) latittude of map center point
    # PARAMETER3:   zoom (int)  map zoom level 0~20
    # RETURN:       the instant of pygmaps
    #========================================================================================
    mymap = pygmaps.maps(spree[15,0], spree[15,1], 16)
    ########## FUNCTION:  addradpoint(latitude, longitude, radius, [color], title)##################
    # DESC:         add a point with a radius (Meter) - Draw cycle
    # PARAMETER1:   latitude (float) latitude of the point
    # PARAMETER2:   longitude (float) longitude of the point
    # PARAMETER3:   radius (float), radius  in meter 
    # PARAMETER4:   color (string) color of the point showed in map, using HTML color code
    #               HTML COLOR CODE:  http://www.computerhope.com/htmcolor.htm
    #               e.g. red "#FF0000", Blue "#0000FF", Green "#00FF00"
    # PARAMETER5:   title (string), label for the point
    # RETURN:       no return 
    #========================================================================================
    mymap.addradpoint(gate[0], gate[1], 95, "#3DFF33")
    mymap.addradpoint(max_pt[1], max_pt[0], 95, "#F433FF")
    mymap.addpoint(max_pt[1], max_pt[0], "#00FFFF")

    ########## FUNCTION:  addpath(path,[color])##############################################
    # DESC:         add a path into map, the data struceture of Path is a list of points
    # PARAMETER1:   path (list of coordinates) e.g. [(lat1,lng1),(lat2,lng2),...]
    # PARAMETER2:   color (string) color of the point showed in map, using HTML color code
    #               HTML COLOR CODE:  http://www.computerhope.com/htmcolor.htm
    #               e.g. red "#FF0000", Blue "#0000FF", Green "#00FF00"
    # RETURN:       no return
    #========================================================================================
    mymap.addpath(spree_path,"#00FFFF")
    mymap.addpath(satt_path, "#4AA02C")
    ########## FUNCTION:  draw(file)######################################################
    # DESC:         create the html map file (.html)
    # PARAMETER1:   file (string) the map path and file
    # RETURN:       no return, generate html file in specified directory
    #========================================================================================
    mymap.draw('cand_loc_gmap.html')

