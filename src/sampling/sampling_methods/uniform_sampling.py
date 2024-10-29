import numpy as np
from ..sampling_constants import SamplingConstants

class UniformSampling:
    
    def generate_samples(number_of_points, dimensionality):
        """
        Generates random uniformly distributed points with the built-in numpy function

        Arguments:
        - number_of_points (int): Number of points the user wants to generate
        - dimensionality (int): Number of dimensions the generated data should have

        Returns:
        - points (ndarray): Generated points
        """
        points = np.random.uniform(SamplingConstants.INTERVAL_LOWER_BOUND, SamplingConstants.INTERVAL_UPPER_BOUND, (number_of_points, dimensionality))
        return points