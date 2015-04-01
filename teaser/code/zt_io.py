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


def plot_gmap(max_pt, gate_pt, spree, satt):

    ##for i, j in xrange(len(spree)):
      ##  spree_path = [(spree_x
    ##spree_path = [(spree[i], spree[j]) for i,j in range(len(spree)) ] 
    ########## CONSTRUCTOR: pygmaps.maps(latitude, longitude, zoom) ##############################
    # DESC:         initialize a map  with latitude and longitude of center point  
    #               and map zoom level "15"
    # PARAMETER1:   latitude (float) latittude of map center point
    # PARAMETER2:   longitude (float) latittude of map center point
    # PARAMETER3:   zoom (int)  map zoom level 0~20
    # RETURN:       the instant of pygmaps
    #========================================================================================
    center_x = 9.30080283
    center_y = 5.71331901
    mymap = pygmaps.maps(center_x, center_y, 16)


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
    mymap.addradpoint(gate_pt[0], gate_pt[1], 95, "#FF0000")#,"Brandenburger Tor")
    mymap.addradpoint(max_pt[0], max_pt[1], 95, "#FF0000") #"candidate's location")


    ########## FUNCTION:  addpath(path,[color])##############################################
    # DESC:         add a path into map, the data struceture of Path is a list of points
    # PARAMETER1:   path (list of coordinates) e.g. [(lat1,lng1),(lat2,lng2),...]
    # PARAMETER2:   color (string) color of the point showed in map, using HTML color code
    #               HTML COLOR CODE:  http://www.computerhope.com/htmcolor.htm
    #               e.g. red "#FF0000", Blue "#0000FF", Green "#00FF00"
    # RETURN:       no return
    #========================================================================================
    #path = [(37.429, -122.145),(37.428, -122.145),(37.427, -122.145),(37.427, -122.146),(37.427, -122.146)]

    mymap.addpath(spree,"#1569C7")
    #mymap.addpath(satt_path,"#6960EC")

    ########## FUNCTION:  draw(file)######################################################
    # DESC:         create the html map file (.html)
    # PARAMETER1:   file (string) the map path and file
    # RETURN:       no return, generate html file in specified directory
    #========================================================================================
    mymap.draw('output/cand_loc.html')

