import numpy as np
from sklearn.preprocessing import normalize
from ..sampling_constants import SamplingConstants

class FibonacciSampling:
    
    def generate_samples(number_of_points, dimensionality):
        """
        Generates points with the Fibonacci Sampling algorithm 
        taken from https://isas.iar.kit.edu/pdf/Fusion21_Frisch.pdf

        Arguments:
        - number_of_points (int): Number of points the user wants to generate
        - dimensionality (int): Number of dimensions the generated data should have

        Returns:
        - XFib_final (ndarray): Generated points
        """
        L = number_of_points
        D = dimensionality
        # Fibonacci matrix eigenvectors
        VD = FibonacciSampling.fib_eigen(D)
        # Smallest hyperrectangle
        s = np.sum(np.abs(VD), axis=0)
        # Smallest hypercube
        sHC = np.max(s)
        # Creating sampling vector r
        L0 = int(np.ceil(L**(1/D)))
        delta = 1.0 / L0
        L1 = int(np.ceil(sHC/delta))
        L1 = L1 + 2
        if L % 2 != L1 % 2:
            L1 = L1 + 1
        r = np.arange(L1) * delta
        centering = np.dot(r, np.ones(L1)) / L1 
        r_centered = r - centering
        # Creating a grid with L1^D points
        Xreg = np.zeros((L, D))
        for i in range(L):
            index = i
            for j in range(D):
                index_j = index % L1
                Xreg[i, j] = r_centered[index_j]
                index //= L1
        Xrot = np.dot(Xreg, VD.T)
        # Removing unwanted points (1)
        is_within_bounds = (Xrot[:, 1:] >= -0.5) & (Xrot[:, 1:] <= 0.5)
        all_conditions_met = np.all(is_within_bounds, axis=1)    
        XFib = Xrot[all_conditions_met]   
        # Sorting by the first coordinate
        sorted_ind = np.argsort(XFib[:, 0])
        XFib_sorted = XFib[sorted_ind]
        # Removing unwanted points (2)
        u = (XFib_sorted.shape[0] - L) // 2
        XFib_final = XFib_sorted[u : u + L]
        # Rescaling for wanted border
        b = 0.5 - 1 / (2*L)
        m = np.max(np.abs(Xrot), axis=0)
        XFib_final = (Xrot / m) * b 
        # Scaling from (-0.5, 0.5) to the right interval
        scaling_factor = (SamplingConstants.INTERVAL_UPPER_BOUND - (SamplingConstants.INTERVAL_LOWER_BOUND)) / (0.5 - (-0.5))
        XFib_final = scaling_factor * (XFib_final + 0.5) - 10        
        return XFib_final
    
    def fib_eigen(D):
        """
        Generates eigenvector matrix with dimension D

        Arguments:
        - D (int): Number of dimensions

        Returns:
        - VD_normalized (ndarray): Normalized eigenvector matrix
        """
        VuD = np.zeros((D, D))
        for i in range(D):
            for j in range(D):
                VuD[i, j] = np.cos(((2*i - 1) * (2*j - 1) * np.pi) / ((4*D) + 2))
        VD_normalized = normalize(VuD, axis=0, norm='l1')
        return VD_normalized