import numpy as np
from ..sampling_constants import SamplingConstants

class GaussianSampling:
            
    def generate_samples(number_of_points, dimensionality):
        """
        Generates random normal distributed points with the built-in numpy function

        Arguments:
        - number_of_points (int): Number of points the user wants to generate
        - dimensionality (int): Number of dimensions the generated data should have

        Returns:
        - points (ndarray): Generated points scaled to the right interval
        """
        generated_points = np.random.normal(SamplingConstants.GAUSSIAN_MEAN, SamplingConstants.GAUSSIAN_STD_DEV, (number_of_points, dimensionality))
        points = np.clip(generated_points, SamplingConstants.INTERVAL_LOWER_BOUND, SamplingConstants.INTERVAL_UPPER_BOUND)
        return points