from projection.projection_methods.pca_projection import PCA_Projection
from projection.projection_methods.tsne_projection import TSNE_Projection
from projection.projection_methods.umap_projection import UMAP_Projection
from projection.projection_methods.axis_aligned_projection import AXIS_ALIGNED_Projection

class ProjectionGenerator:
    
    def generate_visualizations(samples):
        """
        Generates Graph objects for PCA, t-SNE, UMAP and axis-aligned visualizations of the given samples in 1D, 2D and SPLOM

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - tuple: Tuple of all genereated graphs 
        """
        pca_graph_1d = PCA_Projection.generate_1d_visualization(samples)
        tsne_graph_1d = TSNE_Projection.generate_1d_visualization(samples)
        umap_graph_1d = UMAP_Projection.generate_1d_visualization(samples)
        axis_graph_1d = AXIS_ALIGNED_Projection.generate_1d_visualization(samples)
        pca_graph_2d = PCA_Projection.generate_2d_visualization(samples)
        tsne_graph_2d = TSNE_Projection.generate_2d_visualization(samples)
        umap_graph_2d = UMAP_Projection.generate_2d_visualization(samples)
        axis_graph_2d = AXIS_ALIGNED_Projection.generate_2d_visualization(samples)
        pca_graph_splom = PCA_Projection.generate_splom_visualization(samples)
        tsne_graph_splom = TSNE_Projection.generate_splom_visualization(samples)
        umap_graph_splom = UMAP_Projection.generate_splom_visualization(samples)
        axis_graph_splom = AXIS_ALIGNED_Projection.generate_splom_visualization(samples)
        return pca_graph_1d, tsne_graph_1d, umap_graph_1d, axis_graph_1d, pca_graph_2d, tsne_graph_2d, umap_graph_2d, axis_graph_2d, pca_graph_splom, tsne_graph_splom, umap_graph_splom, axis_graph_splom