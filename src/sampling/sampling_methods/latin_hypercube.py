from pyDOE import lhs
from ..sampling_constants import SamplingConstants

class LatinHypercube:
    
    def generate_samples(number_of_points, dimensionality):
        """
        Generates points with Latin Hypercube with the built-in pyDOE function

        Arguments:
        - number_of_points (int): Number of points the user wants to generate
        - dimensionality (int): Number of dimensions the generated data should have

        Returns:
        - points (ndarray): Generated points scaled to the right interval
        """
        generated_points = lhs(dimensionality, samples=number_of_points)
        points = SamplingConstants.INTERVAL_LOWER_BOUND + (SamplingConstants.INTERVAL_UPPER_BOUND - SamplingConstants.INTERVAL_LOWER_BOUND) * generated_points
        return points