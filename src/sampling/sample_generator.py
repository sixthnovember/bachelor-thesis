from sampling.sampling_methods.lattice_sampling import LatticeSampling
from sampling.sampling_methods.latin_hypercube import LatinHypercube
from sampling.sampling_methods.uniform_sampling import UniformSampling
from sampling.sampling_methods.gaussian_sampling import GaussianSampling
from sampling.sampling_methods.fibonacci_sampling import FibonacciSampling
from sampling.sampling_methods.rank1_lattice import Rank1Lattice
from dataframe.dataframe_converter import DataFrameConverter
from sampling.quality_measures import QualityMeasures

class SampleGenerator:
    
    def generate_samples(sampling_method, dimensionality, number_of_points):
        """
        Generates samples with the selected sampling method

        Arguments:
        - sampling_method (str): Name of the used sampling method
        - dimensionality (int): Number of dimensions the generated data should have
        - number_of_points (int): Number of points to generate

        Returns:
        - df (pd.DataFrame): Generated samples
        """
        match sampling_method:
            case 'lattice':
                samples = LatticeSampling.generate_samples(number_of_points, dimensionality)
            case 'hypercube':
                samples = LatinHypercube.generate_samples(number_of_points, dimensionality)
            case 'uniform':
                samples = UniformSampling.generate_samples(number_of_points, dimensionality)
            case 'gaussian':
                samples = GaussianSampling.generate_samples(number_of_points, dimensionality)
            case 'fibonacci':
                samples = FibonacciSampling.generate_samples(number_of_points, dimensionality)
            case 'rank1':
                samples = Rank1Lattice.generate_samples(number_of_points, dimensionality)
        df = DataFrameConverter.convert_to_df(samples, dimensionality)
        return df
    
    def calculate_quality_measures(samples):
        """
        Calculates three quality measures (discrepancy, closest neighbor and max-min-neighbour) for the given samples

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - tuple: A tuple with three strings for the quality measures to display on the website 
        """
        discrepancy = round(QualityMeasures.discrepancy_value(samples), 5)
        closest_neighbor_percent = round(QualityMeasures.closest_neighbor(samples), 2)
        max_min_neighbour = round(QualityMeasures.max_min_neighbour(samples), 2)
        return f'Discrepancy: {discrepancy}', f'Closest Neighbor: {closest_neighbor_percent}%', f'Max-Min Neighbor: {max_min_neighbour}'