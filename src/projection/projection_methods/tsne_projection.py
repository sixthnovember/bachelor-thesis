from sklearn.manifold import TSNE
from projection.graph_generator import GraphGenerator

class TSNE_Projection:
    
    def generate_1d_visualization(samples):
        """
        Uses t-SNE with n_components=1 on the given data and creates a 1D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        tsne_1d = TSNE(n_components=1)
        X_tsne_1d = tsne_1d.fit_transform(samples)
        graph = GraphGenerator.generate_1d_graph(X_tsne_1d, 't-SNE')
        return graph
        
    def generate_2d_visualization(samples):
        """
        Uses t-SNE with n_components=2 on the given data and creates a 2D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        tsne_2d = TSNE(n_components=2)
        X_tsne_2d = tsne_2d.fit_transform(samples)
        graph = GraphGenerator.generate_2d_graph(X_tsne_2d, 't-SNE')
        return graph
    
    def generate_splom_visualization(samples):
        """
        Uses t-SNE with n_components=3 on the given data and creates a SPLOM visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        tsne_splom = TSNE(n_components=3)
        X_tsne_splom = tsne_splom.fit_transform(samples)
        graph = GraphGenerator.generate_splom_graph(X_tsne_splom, 't-SNE')
        return graph