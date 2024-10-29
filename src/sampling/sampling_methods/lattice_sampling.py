from ..sampling_constants import SamplingConstants
import numpy as np

class LatticeSampling:
    
    def generate_samples(number_of_points, dimensionality):
        """
        Generates samples by first creating a grid of points depending on the number of points and dimensionality,
        on which the points are placed evenly 

        Arguments:
        - number_of_points (int): Number of points the user wants to generate
        - dimensionality (int): Number of dimensions the generated data should have

        Returns:
        - points (ndarray): Generated points of shape (number_of_points, dimensionality)
        """
        number_of_points_per_dimension = int(round(number_of_points ** (1/dimensionality)))
        number_of_points_total = number_of_points_per_dimension ** dimensionality
        class_width = (SamplingConstants.INTERVAL_UPPER_BOUND - SamplingConstants.INTERVAL_LOWER_BOUND) / number_of_points_per_dimension
        points_per_dimension = np.linspace(SamplingConstants.INTERVAL_LOWER_BOUND + class_width / 2, SamplingConstants.INTERVAL_UPPER_BOUND - class_width / 2, number_of_points_per_dimension)
        grid = np.meshgrid(*([points_per_dimension] * dimensionality))
        points = np.stack(grid, axis=-1).reshape(-1, dimensionality)
        if number_of_points_total > number_of_points:
            points = points[:number_of_points, :]
        return points