
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

