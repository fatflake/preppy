## Utility functions for zt_main.py
###################################

from scipy.special import erfinv
from scipy.stats import lognorm
import numpy as np
from scipy import optimize

# io imports
from zt_io import read_data, plot_individual

#from proj_coords import project_data

## Constants
############

# Set (0,0) for ROI:
SW_LAT = 52.464011 # Latitude, Y
SW_LON = 13.274099 # Longitude, X
IDX_LON = 1    
IDX_LAT = 0
X_MIN, X_MAX = 0., 20.
Y_MIN, Y_MAX = -5., 15.

CONFIDENCE = .95
m2km = 0.001
SIG2_SPREE = 2730 * m2km
SIG2_SATT  = 2400 * m2km
MEAN_GATE  = 4700 * m2km
MODE_GATE  = 3877 * m2km

# read data
spree, gate, sattelite = read_data()

from proj_coords import project_data

# project data 
def project_data(data):
    """
    Orthogonal projection of GPS data coordinates, gps2xy
    """
    data_x = -(data[:,1] - SW_LON) * np.cos(SW_LAT) * 111.323
    data_y = (data[:,0] - SW_LAT) * 111.323
    return data_x, data_y

SPREE_X, SPREE_Y = project_data(spree)
GATE_X, GATE_Y = project_data(gate)
GATE_X, GATE_Y = GATE_X[0], GATE_Y[0]
SATT_X, SATT_Y = project_data(sattelite)





## Functions
############

def index2xy(index, res):
    """
    From indices in the *res*x*res* space, project to (x,y) coordinates 
    """
    p_x = X_MIN + (index[0] / float(res)) * (X_MAX - X_MIN)
    p_y = Y_MIN + (index[1] / float(res)) * (Y_MAX - Y_MIN)
    return [p_x, p_y]

def xy2gps(points):
    """
    From orthogonally projected (x,y) points, project back to GPS coordinates, xy2GPS
    """
    p_lon = -( points[0] / (111.323*np.cos(SW_LAT)) ) + SW_LON
    p_lat = (points[1] / 111.323) + SW_LAT
    return [p_lon, p_lat]

def compute_line(x_coords, y_coords):
    """
    Computes line between points *x_coords* and *y_coords*, outputs line parameters
    """
    slope = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
    y_int = y_coords[0] - slope*x_coords[0]
    return slope, y_int

def compute_nearest_point(new_point, slope, y_int): 
    """
    Compute nearest_point to *new_point* on line defined by *new_point*, *slope*, *y_int*.
    """
    x_nearest = (new_point[0] + slope*new_point[1] - y_int*slope) / (1 + slope**2)
    y_nearest = (new_point[0]*slope + slope**2*new_point[1] + y_int) / (1 + slope**2)
    nearest_point = np.array([x_nearest, y_nearest])
    return nearest_point


## Probability computations ##

def compute_gauss_sigma(x, confidence):
    """
    Use the inverse ERF of Gaussian to compute the sigma of a Gaussian distribution 
    with confidence interval of size +- *x* around mean 
    containing *confidence* proportion of probability mass
    (Alternatively, you can use a look-up table approximation: http://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule )
    Multiply result (meters) with .001 to have kilometers
    """
    sigma = x / (np.sqrt(2) * erfinv(confidence))
    return sigma

def compute_logn_params(mean, mode):    
    """
    Solution derived using equations...
    mean = exp(mu + sigma**2/2)       
    mode = exp(mu - sigma**2)
    where is mu of log normal is location
    Multiply result (meters) with .001 to have kilometers
    """
    sigma = np.sqrt((2./3) * (np.log(mean)-np.log(mode) ))
    mu = (2./3) * np.log(mean) + (1./3)*np.log(mode)
    return sigma, mu

def gauss_pdf(x_array, mu, sigma):
    """
    Gives pdf of value *x_array* from Gaussian with parameters *mu* and *sigma*.
    """
    prob = (1. / np.sqrt(2*np.pi * sigma**2)) *  (np.exp(-0.5 * (x_array - mu) * (x_array - mu) / sigma**2))
    return prob

def log_gauss_pdf(x_array, mu, sigma):
    #prob = (1. / np.sqrt(2*np.pi * sigma**2)) *  (np.exp(-0.5 * (x_array - mu) * (x_array - mu) / sigma**2))

    """
    For value *x_array* from log-Gaussian with parameters *mu* and *sigma*, outputs pdf value.
    """
    log_prob = -np.log(2 * np.sqrt(sigma*np.pi)) - ( (x_array - mu)**2 / (2* sigma**2) )
    return log_prob


# compute the parameters of the three distributions
print "-" * 70
SIGMA_SPREE = compute_gauss_sigma(SIG2_SPREE, CONFIDENCE)
SIGMA_SATT = compute_gauss_sigma(SIG2_SATT, CONFIDENCE)
SIGMA_GATE, MU_GATE = compute_logn_params(MEAN_GATE, MODE_GATE)
print "Distribution parameters:"
print '- sigma_spree:',SIGMA_SPREE
print '- sigma_satt:',SIGMA_SATT
print '- sigma_gate:',SIGMA_GATE
print '- mu_gate:', MU_GATE


## Spree functions ## 

def compute_spree_projected_pt(new_point, SIGMA_SPREE): 
    """
    Finds the closest point on the Spree path to *new_point* by minimizing 
    the l2-norm of the projected point on all Spree regions.
    """
    X_coords = SPREE_X
    Y_coords = SPREE_Y
    num_segs = len(SPREE_X) - 1
    point_distances = np.zeros(num_segs)
    nearest_points = np.zeros(shape=(num_segs,2))
    # figure out which line segment the new_point is in:
    for i in xrange(num_segs):
        seg_p1 = np.array([X_coords[i], Y_coords[i]])
        seg_p2 = np.array([X_coords[i+1], Y_coords[i+1]])
        # compute line segment between current points:
        slope, y_int = compute_line([seg_p1[0], seg_p2[0]], [seg_p1[1], seg_p2[1]])
        # compute nearest point on that segment to our new point
        nearest_point = compute_nearest_point(new_point, slope, y_int)
        # make sure this point isn't outside the segment
        if nearest_point[0] < seg_p1[0] and nearest_point[0] < seg_p2[0]:
            # check which x_coord is smaller. p1's or p2's, and make nearest pt that coord
            if seg_p1[0] < seg_p2[0]:
                nearest_point = seg_p1
            else:
                nearest_point = seg_p2
        elif nearest_point[0] > seg_p1[0] and nearest_point[0] > seg_p2[0]:
            if seg_p1[0] > seg_p2[0]:
                nearest_point = seg_p1
            else:
                nearest_point = seg_p2
        # collect distance between this nearest point and our new point
        point_distances[i] = np.linalg.norm(new_point - nearest_point)
        # save the nearest point to the list, for later use 
        nearest_points[i,:] = np.array(nearest_point).copy()  
    # find the index of the nearest point of all nearest_points in all segments
    idx_nearest_point = point_distances.argmin()    
    return nearest_points[idx_nearest_point]

def compute_spree_prob(new_point, SIGMA_SPREE): 
    """
    Computes probability of *new_point* along path of Spree, using distance 
    between it and nearest_point as Spree Gauss mean.
    """
    nearest_point = compute_spree_projected_pt(new_point, SIGMA_SPREE)
    dist = np.linalg.norm(new_point - nearest_point)
    prob_of_new_point = log_gauss_pdf(dist, 0, SIGMA_SPREE)
    return prob_of_new_point

## Gate functions ##

def loglognorm(x, mu, sigma):
    """
    Compute log log-Normal pdf.
    """
    log_prob = -np.log(sigma * np.sqrt(2*np.pi)) - np.log(x) - ( (np.log(x) - mu)**2 / (2*sigma**2) )
    return log_prob

def compute_gate_prob(new_point, gate_point, sigma_gate, mean_gate):  
    """
    Compute probability of *new_point* w.r.t. log-normal around *gate_point*.
    """
    dist = np.linalg.norm(new_point - gate_point)
    log_prob = loglognorm(dist, mean_gate, sigma_gate)
    return log_prob


## Sattelite functions ##

def compute_satt_projected_pt(new_point, sigma_satt): 
    """
    Finds the closest point on the Sattelite path to *new_point*.
    """
    slope, y_int = compute_line(SATT_X, SATT_Y)
    nearest_point = compute_nearest_point(new_point, slope, y_int)
    return nearest_point

def compute_satt_prob(new_point, sigma_satt): 
    """
    Computes probability of *new_point* along path of Sattelite.
    """
    mean_satt = 0.
    nearest_point = compute_satt_projected_pt(new_point, sigma_satt)
    dist = np.linalg.norm(new_point - nearest_point) 
    prob_of_new_point = log_gauss_pdf(dist, mean_satt, sigma_satt)
    return prob_of_new_point

## Gradients ##

def gauss_gradient(x_dist, mu, sigma):
    """
    Spree and Sattelite:
    Compute gradient of a log Gaussian probability -- dL(lnN(x_{spr or sa}))/dx_{spr or sa} -- to use in objective function.
    """
    grad = (x_dist - mu) / sigma**2
    return grad

def lognorm_gradient(x_dist, mu, sigma):
    """
    Gate:
    Compute gradient of a log log-Normal probability -- dL(lnN(x_{g}))/dx_{g} -- to use in objective function.
    """
    grad = - (1./x_dist) -  ((np.log(x_dist) - mu) / (x_dist*sigma**2))
    return grad

def point_grad(eval_pt, nearest_pt):
    """
    Compute dx_{g,spr,or sa}/dq_x and dx_{g,spr,or sa}/dq_y to use in objective function.
    """
    grad_x = (eval_pt[0] - nearest_pt[0]) / (np.linalg.norm(eval_pt - nearest_pt))
    grad_y = (eval_pt[1] - nearest_pt[1]) / (np.linalg.norm(eval_pt - nearest_pt))
    return np.array([grad_x, grad_y])


def neg_joint_log_prob(eval_pt):
    """
    Objective function: log joint probability using all out information, f(x)
    """
    x, y = eval_pt[0], eval_pt[1]
    spree_prob = compute_spree_prob(np.array([x,y]), SIGMA_SPREE)
    gate_prob  = compute_gate_prob(np.array([x,y]),  [GATE_X, GATE_Y], SIGMA_GATE, MU_GATE)
    satt_prob  = compute_satt_prob(np.array([x,y]), SIGMA_SATT)
    joint_prob = spree_prob + gate_prob + satt_prob
    return -joint_prob

def neg_joint_log_grad(eval_pt):
    """
    Gradient of objective function: log joint probability, do coordinate descent on to find min of, f'(x).
    """
    eval_pt = np.array(eval_pt)
    mu_spree = 0.
    mu_satt = 0.
    gate_pt = [GATE_X, GATE_Y]
    # Spree:
    spree_nearest_pt = compute_spree_projected_pt(eval_pt, SIGMA_SPREE)
    x_spree = np.linalg.norm(spree_nearest_pt - eval_pt)
    spree_grad = -gauss_gradient(x_spree, mu_spree, SIGMA_SPREE) * point_grad(eval_pt, spree_nearest_pt)
    # Satt:
    satt_nearest_pt = compute_satt_projected_pt(eval_pt, SIGMA_SATT)
    x_satt = np.linalg.norm(satt_nearest_pt - eval_pt)
    satt_grad  = -gauss_gradient(x_satt, mu_satt, SIGMA_SATT) * point_grad(eval_pt, satt_nearest_pt)
    # Gate:
    gate_dist = np.linalg.norm(eval_pt - gate_pt)
    gate_grad  = lognorm_gradient(gate_dist, MU_GATE, SIGMA_GATE) * point_grad(eval_pt, gate_pt)
    # Joint:
    joint_grad = spree_grad + satt_grad + gate_grad    
    return -joint_grad


## Solvers ## 

def compute_grad_desc():
    """
    1) Solve the problem by gradient descent on the objective function (neg joint log probability), (-$, but no visualization).
    """
    # random initialization
    random_x = np.random.random_sample()*(X_MAX-X_MIN) + X_MIN
    random_y = np.random.random_sample()*(Y_MAX-Y_MIN) + Y_MIN
    init_pt = np.array([random_x, random_y]) 
    print "Random initialization:", init_pt
    # wants f(x) and f'(x) as args:
    opt_result = optimize.minimize(neg_joint_log_prob, init_pt, jac=neg_joint_log_grad, method='BFGS')
    max_pt = opt_result.x
    return max_pt


def compute_probs(RES):
    """
    2) Solve the problem using discretezation of the Region of Interest into a 2D grid,
    computing the probability of the candidate at every point ($$$, but gives nice visualization!).
    """
    # region of Berlin for which we care to calculate the probability of candidate
    X = np.linspace(X_MIN, X_MAX, RES)
    Y = np.linspace(Y_MIN, Y_MAX, RES)

    # compute probabilities
    spree_probs = np.zeros( shape=(len(X), len(Y)) )
    gate_probs = np.zeros( shape=(len(X), len(Y)) )
    satt_probs = np.zeros( shape=(len(X), len(Y)) )
    prob_max = 0.0
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            # calculate all probabilities in log space, not probability space:
            spree_prob = compute_spree_prob(np.array([x,y]), SIGMA_SPREE)
            gate_prob  = compute_gate_prob(np.array([x,y]),  [GATE_X, GATE_Y], SIGMA_GATE, MU_GATE)#MEAN_GATE)
            satt_prob  = compute_satt_prob(np.array([x,y]), SIGMA_SATT)
            spree_probs[i,j] = spree_prob
            gate_probs[i,j]  = gate_prob
            satt_probs[i,j]  = satt_prob

    plot_individual(spree_probs, gate_probs, satt_probs)
    probs = spree_probs +  satt_probs  + gate_probs        
    return probs

