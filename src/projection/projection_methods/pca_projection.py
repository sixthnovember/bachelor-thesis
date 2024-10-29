from sklearn.decomposition import PCA
from projection.graph_generator import GraphGenerator

class PCA_Projection:
    
    def generate_1d_visualization(samples):
        """
        Uses PCA with n_components=1 on the given data and creates a 1D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        pca_1d = PCA(n_components=1)
        X_pca_1d = pca_1d.fit_transform(samples) 
        graph = GraphGenerator.generate_1d_graph(X_pca_1d, 'PCA')
        return graph

    def generate_2d_visualization(samples):
        """
        Uses PCA with n_components=2 on the given data and creates a 2D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(samples)
        graph = GraphGenerator.generate_2d_graph(X_pca_2d, 'PCA')
        return graph
    
    def generate_splom_visualization(samples):
        """
        Uses PCA with n_components=3 on the given data and creates a SPLOM visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        pca_splom = PCA(n_components=3)
        X_pca_splom = pca_splom.fit_transform(samples)
        graph = GraphGenerator.generate_splom_graph(X_pca_splom, 'PCA')
        return graph