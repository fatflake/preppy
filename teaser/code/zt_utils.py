## Utility functions for zt_main.py
###################################

from scipy.special import erfinv
from scipy.stats import lognorm
import numpy as np
import math
from math import cos

# io imports
from zt_io import read_data

from proj_coords import project_data

## Constants
############

# Set (0,0) for ROI:
SW_LAT = 52.464011 # Latitude
SW_LON = 13.274099 # Longitude
IDX_LAT = 0
IDX_LON = 1

CONFIDENCE = .95
SIG2_SPREE = 2730
SIG2_SATT  = 2400
MEAN_GATE  = 4700
MODE_GATE  = 3877

# read data
spree, gate, sattelite = read_data()


# project data 
def project_data(data):
    """
    Orthogonal projection of GPS data coordinates
    """
    data_x = -(data[:,IDX_LON] - SW_LON) * cos(SW_LAT) * 111.323
    data_y = (data[:,IDX_LAT] - SW_LAT) * 111.323
    return data_x, data_y

SPREE_X, SPREE_Y = project_data(spree)
GATE_X, GATE_Y = project_data(gate)
SATT_X, SATT_Y = project_data(sattelite)


## Functions
############

def compute_gauss_sigma(x, confidence):
    """
    Use the inverse ERF of Gaussian to compute the sigma of a Gaussian distribution 
    with confidence interval of size +- *x* around mean 
    containing *confidence* proportion of probability mass
    (Alternatively, you can use a look-up table approximation: http://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule )
    """
    sigma_m = x / (np.sqrt(2) * erfinv(confidence))
    # result is in m, convert to km to use  in rest of code
    sigma = sigma_m * 0.001
    return sigma

def compute_logn_sigma(mean, mode):    
    """
    Solution derived using equations...
    mean = exp(mu + sigma**2/2)       
    mode = exp(mu - sigma**2)
    where is mu of log normal is location
    """
    sigma_m = np.sqrt( mode - np.log(mean) )
    sigma = sigma_m * 0.001
    return sigma

def gauss_pdf(x_array, mu, sigma):
    prob = (1. / math.sqrt(2*math.pi * sigma**2)) *  (np.exp(-0.5 * (x_array - mu) * (x_array - mu) / sigma**2))
    return prob

def compute_line(x_coords, y_coords):
    slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    y_int = y_coords[0] - slope*x_coords[0]
    return slope, y_int

def compute_nearest_point(new_point, slope, y_int): 
    x_nearest = (new_point[0] + slope*new_point[1] - y_int*slope) / (1 + slope**2)
    y_nearest = (new_point[0]*slope + slope**2*new_point[1] + y_int) / (1 + slope**2)
    nearest_point = [x_nearest, y_nearest]
    return nearest_point

def compute_spree_prob(new_point, sigma_spree): 
    X_coords = SPREE_X
    Y_coords = SPREE_Y
    num_segs = len(SPREE_X) - 1
    point_distances = np.zeros(num_segs)
    # figure out which line segment the new_point is in:
    for i in xrange(num_segs):
        seg_p1 = [X_coords[i], Y_coords[i]]
        seg_p2 = [X_coords[i+1], Y_coords[i+1]]
        # compute line segment between current points:
        slope, y_int = compute_line([seg_p1[0], seg_p2[0]], [seg_p1[1], seg_p2[1]])
        # compute nearest point on that segment to our new point
        nearest_point = compute_nearest_point(new_point, slope, y_int)
        # make sure this point isn't outside the segment
        # XXX if new_point[0] <= nearest_point[0]:
        if nearest_point[0] < seg_p1[0] and nearest_point[0] < seg_p2[0]:
            # check which x_coord is smaller. p1's or p2's, and make nearest pt that coord
            if seg_p1[0] < seg_p2[0]:
                nearest_point = seg_p1
            else:
                nearest_point = seg_p2
        # XXX elif new_point[0] => nearest_point[0]:
        elif nearest_point[0] > seg_p1[0] and nearest_point[0] > seg_p2[0]:
            if seg_p1[0] > seg_p2[0]:
                nearest_point = seg_p1
            else:
                nearest_point = seg_p2
        # collect distance between this nearest point and our new point
        point_distances[i] = np.linalg.norm(new_point - nearest_point)
    # find the nearest point of all nearest_points in all segments
    idx_nearest_point = point_distances.argmin()    
    # compute the probability of the distance to this point
    prob_of_new_point = gauss_pdf(point_distances[idx_nearest_point], 0, sigma_spree)
    return prob_of_new_point


def compute_gate_prob(new_point, gate_point, sigma_gate, mean_gate):  
    ### FIXME XXX need log normal -- this is the only part I'm not sure I'm doing right

    #log_norm = lognorm([sigma_gata],loc=MODE_GATE) # std, loc=mean
    dist = np.exp(np.log(np.linalg.norm(new_point - gate_point))) # do i want distance?
    prob_of_new_point = lognorm.pdf(dist, np.exp(np.log(sigma_gate)) ) 
    return prob_of_new_point

def wiktors_butt(new_point, sigma_spree):
    X_coords = SPREE_X
    Y_coords = SPREE_Y
    nn_dist = float('inf')
    nn = np.array([0., 0.])
    for k in range(1, len(X_coords)):
        # 'NEW SEGMENT'
        p1 = np.array([X_coords[k-1], Y_coords[k-1]])
        p2 = np.array([X_coords[k], Y_coords[k]])
        m, y0 = compute_line(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]))
        # print 'new_point0:',new_point
        pp = np.array(compute_nearest_point(new_point, m, y0))
        # print 'new_point1:',new_point
        if ((pp[0] < p1[0] and pp[0] < p2[0]) or (pp[0] > p1[0] and pp[0] > p2[0])):
            p1_dist = np.linalg.norm(p1 - new_point)
            p2_dist = np.linalg.norm(p2 - new_point)
            if p1_dist < p2_dist:
                pp = p1
            else:
                pp = p2
        np_dist = np.linalg.norm(pp - new_point)
        if np_dist < nn_dist:
            nn = np.array(pp)
            nn_dist = np_dist
    prob_of_new_point = gauss_pdf(nn_dist, 0.0, sigma_spree)
    return prob_of_new_point

def compute_satt_prob(new_point, sigma_satt): 
    mean_satt = 0.
    slope, y_int = compute_line(SATT_X, SATT_Y)
    nearest_point = compute_nearest_point(new_point, slope, y_int)
    dist = np.linalg.norm(new_point - nearest_point) 
    prob_of_new_point = gauss_pdf(dist, mean_satt, sigma_satt)
    return prob_of_new_point

def compute_probs(RES):
    # region of Berlin for which we care to calculate the probability of candidate
    X_MIN, X_MAX = 0., 20.
    Y_MIN, Y_MAX = -5., 15.
    X = np.linspace(X_MIN, X_MAX, RES)
    Y = np.linspace(Y_MIN, Y_MAX, RES)

    # compute the parameters of the three distributions
    sigma_spree = compute_gauss_sigma(SIG2_SPREE, CONFIDENCE)
    sigma_satt = compute_gauss_sigma(SIG2_SATT, CONFIDENCE)
    sigma_gate = compute_logn_sigma(MEAN_GATE, MODE_GATE)
    print 'sigma spree:',sigma_spree
    print 'sigma satt:',sigma_satt
    print 'sigma gate:',sigma_gate

    # compute probabilities
    probs = np.zeros( shape=(len(X), len(Y)) )
    prob_max = 0.0
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            spree_prob = compute_spree_prob(np.array([x,y]), sigma_spree)
            gate_prob  = compute_gate_prob(np.array([x,y]),  [GATE_X, GATE_Y], sigma_gate, MEAN_GATE)
            #print "gate prob:", gate_prob
            satt_prob  = compute_satt_prob(np.array([x,y]), sigma_satt)
            probs[i,j] = spree_prob * satt_prob * gate_prob # * satt_prob #spree_prob + gate_prob + satt_prob
            ##probs[i,j] = wiktors_butt(np.array([x,y]), sigma_spree)
    
    return probs


