## Fun Zalando Teaser
##
## J. Shelton / 2015
####################

import numpy as np, sys

# io imports
import zt_io

# utils imports
import zt_utils


## Main script
##############

# read data
spree, gate, sattelite = zt_io.read_data()

# project data
spree_x, spree_y = zt_utils.project_data(spree)
gate_x, gate_y = zt_utils.project_data(gate)
sattelite_x, sattelite_y = zt_utils.project_data(sattelite)

# initial plot 
zt_io.plot_everything([spree_x, spree_y], [gate_x, gate_y], [sattelite_x, sattelite_y]) 

print "-" * 40
print "(1) find candidate location using gradient descent on the (neg) log joint probability"
grad_desc_location = zt_utils.compute_grad_desc()
print "Gradient descent solution location:"
print "-- XY coords:", grad_desc_location[::-1]
grad_desc_gps = zt_utils.xy2gps(grad_desc_location)
print "-- GPS:", grad_desc_gps[::-1]

print "-" * 40
print "(2) find candidate location by evaluating 2D input grid"
# set resolution of 2D grid
RES = 100

# compute joint probabilities for grid
probs = zt_utils.compute_probs(RES) 

# take max
max_prob = np.exp(probs.max())
location_idx = np.unravel_index(probs.argmax(), (RES,RES)) 
location_xy = zt_utils.index2xy(location_idx, RES)
location_gps = zt_utils.xy2gps(location_xy)

print "Resolution of discretization: %d x %d" % (RES, RES)
print "Grid solution, min of -(log(probability)):" 
print "-- index:", location_idx[::-1]
print "-- XY coords:", location_xy[::-1]
print "-- GPS:", location_gps[::-1]

# TODO fix plotting to use a coordinate mapping function
# TODO one func t odo solution? learn params only once...
 
# plot/save result
zt_io.plot_joint(probs, location_idx)
#zt_io.plot_joint(probs_xy, location_idx)
zt_io.plot_gmap(grad_desc_gps, gate, spree, sattelite)


