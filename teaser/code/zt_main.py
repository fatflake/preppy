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
RES = 500

# compute probabilities
probs = zt_utils.compute_probs(RES) 

# take max
max_prob = probs.max()
location_idx = np.unravel_index(probs.argmax(), (RES,RES)) 
location_xy = zt_utils.index2xy(location_idx, RES)
location_gps = zt_utils.gps_coords(location_xy)

sameh = np.array([[52.51104008, 13.45584084]])
loc_sameh = zt_utils.project_data(sameh)
gps_sameh = zt_utils.gps_coords(loc_sameh)
print "Sameh's original GPS:", sameh
print "Sameh's solution, XY coords:", loc_sameh, "GPS (recalc'ed) coords:", gps_sameh
print "Our solution, index:", location_idx, "and XY coords:", location_xy, "and GPS:", location_gps

print "Resolution of discretization: %d x %d" % (RES, RES)
print "With probability ", max_prob, " that candidate is at GPS coordinates: ", location_gps

# TODO fix plotting to use a coordinate mapping function

# plot/save result
zt_io.plot_joint(probs, location_idx)


