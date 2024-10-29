from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from projection.graph_generator import GraphGenerator

class AXIS_ALIGNED_Projection:

    def generate_1d_visualization(samples):
        """
        Uses a RandomForestClassifier to find the most important feature of 
        the given data and creates a 1D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        X_train, X_test, y_train, y_test = train_test_split(samples, np.zeros(samples.shape[0]), test_size=0.3, random_state=42)
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_train, y_train)    
        importances = forest.feature_importances_
        top_feature_idx = np.argsort(importances)[-1]
        selected_feature = samples[:, top_feature_idx]
        graph = GraphGenerator.generate_1d_graph(selected_feature, 'axis-aligned')
        return graph

    def generate_2d_visualization(samples):
        """
        Uses a RandomForestClassifier to find the two most important features of 
        the given data and creates a 2D visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        X_train, X_test, y_train, y_test = train_test_split(samples, np.zeros(samples.shape[0]), test_size=0.3, random_state=42)
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_two_features = indices[:2]
        selected_features = samples[:, top_two_features]
        graph = GraphGenerator.generate_2d_graph(selected_features, 'axis-aligned')
        return graph
    
    def generate_splom_visualization(samples):
        """
        Uses a RandomForestClassifier to find the three most important features of 
        the given data and creates a SPLOM visualization of the results

        Arguments:
        - samples (ndarray): Previously generated samples

        Returns:
        - graph (dcc.Graph): Graph that can be displayed on the web page
        """
        X_train, X_test, y_train, y_test = train_test_split(samples, np.zeros(samples.shape[0]), test_size=0.3, random_state=42)
        forest = RandomForestClassifier(n_estimators=100, random_state=42)
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_three_features = indices[:3]
        selected_features = samples[:, top_three_features]
        graph = GraphGenerator.generate_splom_graph(selected_features, 'axis-aligned')
        return graph