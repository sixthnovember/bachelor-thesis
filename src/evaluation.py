import os
from sampling.sample_generator import SampleGenerator
from sklearn.discriminant_analysis import StandardScaler
from sampling.quality_measures import QualityMeasures
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def evaluate(sampling_method, dimensionality, number_of_points): 
    """
    Calculates three quality measures (discrepancy, closest neighbor and max-min-neighbour)
    
    Arguments:
    - sampling_method (str): Name of the sampling method
    - dimensionality (int): Number of dimensions the generated data should have
    - number_of_points (int): Number of points to generate

    Returns:
    - tuple: The input arguments + the three quality measures
    """
    sampled_df = SampleGenerator.generate_samples(sampling_method, dimensionality, number_of_points)
    X = sampled_df.iloc[:, :dimensionality].values
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    discrepancy = round(QualityMeasures.discrepancy_value(X_standardized), 5)
    closest_neighbor_percent = round(QualityMeasures.closest_neighbor(X_standardized), 2)
    max_min_neighbour = round(QualityMeasures.max_min_neighbour(X_standardized), 2)
    return sampling_method, dimensionality, number_of_points, discrepancy, closest_neighbor_percent, max_min_neighbour

def generate_linechart(results, quality_measure, ylabel, title, output_dir):
    """
    Stores the results in a linechart and saves it as an image
    
    Arguments:
    - results (list): Results of the evaluation
    - quality_measure (str): Name of the quality measure
    - ylabel (str): The label for the y-axis
    - title (str): Title of the chart
    - output_dir (str): Directory to save the image to
    """
    df = pd.DataFrame(results, columns=['Method', 'Dimension', 'Points', 'Discrepancy', 'Closest Neighbor', 'Max-Min Neighbor'])
    fig = px.line(df, x='Dimension', y=quality_measure, color='Method', title=title, labels={'Dimension': 'Dimension', quality_measure: ylabel})
    fig.update_layout(legend_title_text='Sampling Method')
    output_file = os.path.join(output_dir, f'{quality_measure.replace(" ", "_")}_Chart.png')
    fig.write_image(output_file)
    print(f"Chart saved as an image at {output_file}")

def generate_table(results, quality_measure, title, output_dir):
    """
    Generates a table from the evaluation results and saves it as an image
    
    Arguments:
    - results (list): Results of the evaluation
    - quality_measure (str): Name of the quality measure
    - title (str): Title of the table
    - output_dir (str): Directory to save the image to
    """
    df = pd.DataFrame(results, columns=['Method', 'Dimension', 'Points', 'Discrepancy', 'Closest Neighbor', 'Max-Min Neighbor'])
    df_pivot = df.pivot(index='Dimension', columns='Method', values=quality_measure)
    df_pivot = df_pivot[['lattice', 'hypercube', 'uniform', 'gaussian', 'fibonacci', 'rank1']]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.set_title(title)
    table = ax.table(cellText=df_pivot.values, colLabels=df_pivot.columns, rowLabels=[f"{dim}D" for dim in df_pivot.index], cellLoc='center', loc='center')
    output_file = os.path.join(output_dir, f'{quality_measure.replace(" ", "_")}_Table.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Table saved as an image at {output_file}")

output_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(output_dir, exist_ok=True)

sampling_methods = ['lattice', 'hypercube', 'uniform', 'gaussian', 'fibonacci', 'rank1']
results = [evaluate(method, dim, points) for method in sampling_methods for dim in [3, 4, 5, 6, 7, 8, 9, 10] for points in [500]]

generate_linechart(results, 'Discrepancy', 'Discrepancy Value', 'Discrepancy for Sampling Methods', output_dir)
generate_linechart(results, 'Closest Neighbor', 'Closest Neighbor (%)', 'Closest Neighbor for Sampling Methods', output_dir)
generate_linechart(results, 'Max-Min Neighbor', 'Max-Min Neighbor Value', 'Max-Min Neighbor for Sampling Methods', output_dir)

generate_table(results, 'Discrepancy', 'Discrepancy for Sampling Methods', output_dir)
generate_table(results, 'Closest Neighbor', 'Closest Neighbor for Sampling Methods', output_dir)
generate_table(results, 'Max-Min Neighbor', 'Max-Min Neighbor for Sampling Methods', output_dir)