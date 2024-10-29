import numpy as np
from ..sampling_constants import SamplingConstants

class Rank1Lattice:
    
    def generate_samples(number_of_points, dimensionality):
        """
        Generates samples with a generateor vector 

        Arguments:
        - number_of_points (int): Number of points the user wants to generate
        - dimensionality (int): Number of dimensions the generated data should have

        Returns:
        - points (ndarray): Generated points
        """
        generator_vector = np.zeros(dimensionality)
        for i in range(dimensionality):
            generator_vector[i] = (SamplingConstants.RANK_1_A**i) % number_of_points
        points = np.zeros((number_of_points, dimensionality))
        for i in range(number_of_points):
            for j in range(dimensionality):
                points[i, j] = (i * generator_vector[j]) % number_of_points
        points = points / number_of_points        
        scaling_factor = (SamplingConstants.INTERVAL_UPPER_BOUND - (SamplingConstants.INTERVAL_LOWER_BOUND)) / (1 - 0)
        points = (points * scaling_factor) - 10
        return points