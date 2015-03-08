## Utility functions for zt_main.py
###################################

from scipy.special import erfinv
import numpy as np
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
    sigma = x / (np.sqrt(2) * erfinv(confidence))
    return sigma

def compute_logn_sigma(mean, mode):    
    """
    Solution derived using equations...
    mean = exp(mu + sigma**2/2)       
    mode = exp(mu - sigma**2)
    where is mu of log normal is location
    """
    sigma = np.sqrt( mode - np.log(mean) )
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

def compute_spree_prob(new_point, sigma_spree): # FIXME

    X_coords = len(SPREE_X)
    Y_coords = len(SPREE_Y) 
    # TODO somehow figure out which line segment in
    for i in xrange(X_coords):
        x_loc = new_point[0] <= SPREE_X[i]      # FIXME need to be careful of when it's =, then its between segments
        if x_loc:
            x_seg = i
        
        #elif new_point[0] == SPREE_X[i]:
          #  nearest point = [SPREE_X[i], SPREE_Y[i]]
        

    # compute_line(), compute_nearest_point()

    return prob_of_new_point


def compute_gate_prob(new_point, gate_point, sigma_gate, mean_gate):  
    dist = np.linalg.norm(new_point - gate_point) 
    ### FIXME XXX need log normal
    prob_of_new_point = gauss_pdf(dist, mean_gate, sigma_gate)
    return prob_of_new_point

def wiktors_butt(new_point, sigma_spree):
    X_coords = SPREE_X
    Y_coords = SPREE_Y
    nn_dist = float('inf')
    nn = np.array([0., 0.])
    for k in range(1, len(SPREE_X)):
        p1 = np.array([X_coords[k-1], Y_coords[k-1]])
        p2 = np.array([X_coords[k], Y_coords[k]])
        m, y0 = compute_line(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]))
        pp = np.array(compute_nearest_point(new_point, m, y0))
        if pp[0] < p1[0] or pp[0] > p2[0]:
            p1_dist = np.linalg.norm(p1 - new_point)
            p2_dist = np.linalg.norm(p2 - new_point)
            if p1_dist < p2_dist:
                pp = p1
            else:
                pp = p2
        np_dist = np.linalg.norm(pp - nn)
        if np_dist < nn_dist:
            nn = pp
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

def compute_probs(resolution):
    X, Y = len(resolution)

    sigma_spree = compute_gauss_sigma(SIG2_SPREE, CONFIDENCE)
    sigma_satt = compute_gauss_sigma(SIG2_SATT, CONFIDENCE)
    sigma_gate = compute_logn_sigma(MEAN_GATE, MODE_GATE)

    # compute probabilities
    probs = np.zeros( shape=(X, Y) )
    for x in xrange(X):
        for y in xrange(Y):            
            spree_prob = compute_spree_prob([x,y], sigma_spree)
            gate_prob  = compute_gate_prob([x,y],  [GATE_X, GATE_Y], sigma_gate, MEAN_GATE)
            satt_prob  = compute_satt_prob([x,y], sigma_satt)
            probs[x,y] = spree_prob * gate_prob * satt_prob

    return probs

