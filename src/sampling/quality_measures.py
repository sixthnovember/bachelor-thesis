from sampling.sampling_constants import SamplingConstants
from scipy.stats.qmc import discrepancy as disc
from scipy.spatial.distance import pdist, squareform
import numpy as np

class QualityMeasures:
    
    def discrepancy_value(points):
        """
        Calculates the discrepancy of the given points with the built-in scipy function

        Arguments:
        - points (ndarray): Previously generated samples

        Returns:
        - discrepancy (float): Calculated discrepancy of the given points
        """
        rescaled_points = (points - SamplingConstants.INTERVAL_LOWER_BOUND) / (SamplingConstants.INTERVAL_UPPER_BOUND - SamplingConstants.INTERVAL_LOWER_BOUND)
        discrepancy = disc(rescaled_points)
        return discrepancy

    def closest_neighbor(points):
        """
        Calculates the closest neighbor in percent of the given points

        Arguments:
        - points (ndarray): Previously generated samples

        Returns:
        - min_neighbor_percent (float): Calculated closest neighbor in percent of the given points
        """
        dist_matrix = squareform(pdist(points))
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        min_neighbor_percent = (np.min(min_distances) / np.max(min_distances)) * 100
        return min_neighbor_percent
    
    def max_min_neighbour(points):
        """
        Calculates the maximum minimum distance of the given points

        Arguments:
        - points (ndarray): Previously generated samples

        Returns:
        - max_nearest_neighbor (float): Maximum minimum distance of the given points
        """
        dist_matrix = squareform(pdist(points))
        np.fill_diagonal(dist_matrix, np.inf)
        min_distances = np.min(dist_matrix, axis=1)
        max_nearest_neighbor = np.max(min_distances)
        return max_nearest_neighbor