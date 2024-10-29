import pandas as pd

class DataFrameGenerator:
    
    def generate_df(X, projection_medthod, dimension):
        """
        Generates a pandas Dataframe from the previously generated samples based on the selected projection method

        Arguments:
        - X (ndarray): Standardized samples
        - projection_medthod (str): Name of the used projection method
        - dimension (int): Number of dimensions of X

        Returns:
        - pd.Dataframe: Dataframe where each column represents a feature and each row a point
        """   
        if projection_medthod == 'PCA':
            if dimension == '1D':
                return pd.DataFrame(X, columns=['Principal Component 1'])
            elif dimension == '2D':
                return pd.DataFrame(X, columns=['Principal Component 1', 'Principal Component 2'])  
            else: 
                columns = []
                num_components = X.shape[1]
                for i in range(num_components):
                    column_name = f'PC {i+1}'
                    columns.append(column_name)
                return pd.DataFrame(X, columns=columns)
        else:
            if dimension == '1D':
                df = pd.DataFrame(X, columns=[projection_medthod + ' Feature 1'])
                df['y'] = 0
                return df
            elif dimension == '2D':
                return pd.DataFrame(X, columns=[projection_medthod + ' Feature 1', projection_medthod + ' Feature 2'])
            else: 
                columns = []
                num_components = X.shape[1]
                for i in range(num_components):
                    column_name = f'Feature {i+1}'
                    columns.append(column_name)
                return pd.DataFrame(X, columns=columns)