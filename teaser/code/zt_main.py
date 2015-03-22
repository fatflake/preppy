## Fun Zalando Teaser
##
## J. Shelton / 2015
####################

import numpy as np

# should these be in the file i import or here?

# io imports
import zt_io

# utils imports
import zt_utils


## Main script
##############

print "-----------------------------------------------------------"
# read data
spree, gate, sattelite = zt_io.read_data()

# project data
spree_x, spree_y = zt_utils.project_data(spree)
gate_x, gate_y = zt_utils.project_data(gate)
sattelite_x, sattelite_y = zt_utils.project_data(sattelite)

# initial plot 
zt_io.plot_everything([spree_x, spree_y], [gate_x, gate_y], [sattelite_x, sattelite_y]) 

# set resolution of 2D input grid
RES = 100

# compute probabilities
probs = zt_utils.compute_probs(RES) 

# take max
max_prob = probs.max()
location = np.unravel_index(probs.argmax(), (RES,RES)) 
print "Resolution of discretization is %d x %d" % (RES, RES)
print "With probability ", max_prob, " that candidate is at point: ", location

# TODO fix plotting to use a coordinate mapping function

# plot/save result
zt_io.plot_joint(probs, location)


