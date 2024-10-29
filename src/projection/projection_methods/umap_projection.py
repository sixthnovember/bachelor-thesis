import umap
from projection.graph_generator import GraphGenerator

class UMAP_Projection:
        
    def generate_1d_visualization(samples):
        """
        Uses UMAP with n_components=1 on the given data and creates a 1D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        umap_1d = umap.UMAP(n_components=1)
        X_umap_1d = umap_1d.fit_transform(samples)
        graph = GraphGenerator.generate_1d_graph(X_umap_1d, 'UMAP')
        return graph
 
    def generate_2d_visualization(samples):
        """
        Uses UMAP with n_components=2 on the given data and creates a 2D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        umap_2d = umap.UMAP(n_components=2)
        X_umap_2d = umap_2d.fit_transform(samples)
        graph = GraphGenerator.generate_2d_graph(X_umap_2d, 'UMAP')
        return graph

    def generate_splom_visualization(samples):
        """
        Uses UMAP with n_components=3 on the given data and creates a SPLOM visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        umap_splom = umap.UMAP(n_components=3)
        X_umap_splom = umap_splom.fit_transform(samples)
        graph = GraphGenerator.generate_splom_graph(X_umap_splom, 'UMAP')
        return graph