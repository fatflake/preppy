## Project coordinates 
######################


from math import cos


def project_data(data):
    """
    Orthogonal projection of GPS data coordinates
    """
    data_x = -(data[:,IDX_LON] - SW_LON) * cos(SW_LAT) * 111.323
    data_y = (data[:,IDX_LAT] - SW_LAT) * 111.323
    return data_x, data_y
