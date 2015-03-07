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

# read data
spree, gate, sattelite = zt_io.read_data()

# project data

spree_x, spree_y = zt_utils.project_data(spree)
gate_x, gate_y = zt_utils.project_data(gate)
sattelite_x, sattelite_y = zt_utils.project_data(sattelite)

# initial plot 
zt_io.plot_everything() #spree_x, spree_y, gate_x, gate_y, sattelite_x, sattelite_y)

# make 2D input grid
resolution = 10 # TODO

# compute probabilities
### spree_prob, gate_prob, satt_prob = 
probs = compute_probs(resolution) ### spree_prob *. gate_prob *. satt_prob # FIXME pointwise mtrix mult??

# take max
location = probs.argmax()



