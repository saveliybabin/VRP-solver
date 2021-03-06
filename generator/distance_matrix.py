#https://codereview.stackexchange.com/questions/98275/compute-spherical-distance-matrix-from-list-of-geographical-coordinates
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def distance_function(coordinate_array, is_coord):
    if is_coord:
        """
        Compute a distance matrix of the coordinates using a spherical metric.
        :param coordinate_array: numpy.ndarray with shape (n,2); latitude is in 1st col, longitude in 2nd.
        :returns distance_mat: numpy.ndarray with shape (n, n) containing distance in km between coords.
        """
        # Radius of the earth in km (GRS 80-Ellipsoid)
        EARTH_RADIUS = 6371.007176 

        # Unpacking coordinates
        latitudes = coordinate_array[:, 0]
        longitudes = coordinate_array[:, 1]

        # Convert latitude and longitude to spherical coordinates in radians.
        degrees_to_radians = np.pi/180.0
        phi_values = (90.0 - latitudes)*degrees_to_radians
        theta_values = longitudes*degrees_to_radians

        # Expand phi_values and theta_values into grids
        theta_1, theta_2 = np.meshgrid(theta_values, theta_values)
        theta_diff_mat = theta_1 - theta_2

        phi_1, phi_2 = np.meshgrid(phi_values, phi_values)

        # Compute spherical distance from spherical coordinates
        angle = (np.sin(phi_1) * np.sin(phi_2) * np.cos(theta_diff_mat) + 
               np.cos(phi_1) * np.cos(phi_2))
        arc = np.arccos(angle)

        # Multiply by earth's radius to obtain distance in km
        return arc * EARTH_RADIUS
    else:
        return pd.DataFrame(squareform(pdist(coordinate_array)))                        