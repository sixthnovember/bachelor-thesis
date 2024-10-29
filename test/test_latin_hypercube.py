import unittest
import numpy as np
from src.sampling.sampling_constants import SamplingConstants
from src.sampling.sampling_methods.latin_hypercube import LatinHypercube

class TestLatinHyperCube(unittest.TestCase):
    
    def setUp(self):
        """
        Generate samples for testing
        """
        self.number_of_points = 500
        self.dimensionality = 5
        self.samples = LatinHypercube.generate_samples(self.number_of_points, self.dimensionality)
    
    def test_generate_samples_shape(self):
        """
        Test if the shape of the samples is equal to (number_of_points, dimensionality)
        """
        self.assertEqual(self.samples.shape, (self.number_of_points, self.dimensionality), "The shape of the samples should be equal to (number_of_points, dimensionality).")
        
    def test_generate_samples_interval_bounds(self):
        """
        Test if the generated samples are within the given interval
        """
        self.assertGreaterEqual(np.min(self.samples), SamplingConstants.INTERVAL_LOWER_BOUND, "Some samples are below the lower bound.")
        self.assertLessEqual(np.max(self.samples), SamplingConstants.INTERVAL_UPPER_BOUND, "Some samples are above the upper bound.")

if __name__ == "__main__":
    unittest.main()