import pandas as pd

class DataFrameConverter:
    
    def convert_to_df(points, dimensionality):
        """
        Converts the points from an ndarray to a pandas Dataframe

        Arguments:
        - points (ndarray): ndarray of floating point numbers
        - dimensionality (int): Dimension of the points

        Returns:
        - pd.Dataframe: Dataframe where each column represents the dimension and each row a point
        """
        columns = []
        for i in range(dimensionality):
            columns.append(f'{i+1}D')
        return pd.DataFrame(points, columns=columns)