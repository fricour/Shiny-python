import yaml
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the YAML configuration
def load_yaml_config(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None

# Function to load the NetCDF file
def load_netcdf(result_dir, simulation_name, setup, station, is_rates=False):
    file_name = f'{station}_result_rates.nc' if is_rates else f'{station}_result.nc'
    file_path = os.path.join(result_dir, simulation_name, setup, file_name)
    if os.path.exists(file_path):
        return xr.open_dataset(file_path)
    return None

# Function to plot a Taylor diagram
def create_target_diagram(df, add_annotations=False):
    """
    Create a target diagram based on the provided statistics.
    
    Args:
    df (pd.DataFrame): DataFrame containing the statistics
    add_annotations (bool): Whether to add text annotations to the plot
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """

    # compute some additional statistics
    df['NormalizedBias']  = df['MeanBias'] / df['StandardDeviationObservations']
    df['NormalizedcRMSD'] = df['UnbiasedRMSE'] / df['StandardDeviationObservations'] * np.sign(df['StandardDeviationModel'] - df['StandardDeviationObservations'] )

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create the scatter plot
    plot = sns.scatterplot(data=df, x="NormalizedcRMSD", y='NormalizedBias', 
                    hue='variable', style = 'setup', s=150, ax=ax, legend=True)
    
    # Add vertical and horizontal lines
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # Add circular guidelines
    for radius in [0.5, 1]:
        circle = plt.Circle((0, 0), radius, color='black', fill=False, 
                            linestyle='--' if radius == 0.5 else '-')
        ax.add_patch(circle)
    
    # Add variable labels to each point
    for _, row in df.iterrows():
        ax.annotate(row['variable'], (row["NormalizedcRMSD"], row["NormalizedBias"]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Set plot limits
    #ax.set_xlim([-3, 3])
    #ax.set_ylim([-3, 3])
    
    # Add title and labels
    #ax.set_title("Target Diagram", fontsize=16)
    ax.set_xlabel("Normalized cRMSD", fontsize=12)
    ax.set_ylabel("Normalized Bias", fontsize=12)

    # Add legend
    plot.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    
    if add_annotations:
        annotations = [
            (0, 1.8, 'Model overestimates', 'center', 'center'),
            (0, -1.8, 'Model underestimates', 'center', 'center'),
            (1.8, 0, 'Too much \nvariations in model', 'center', 'center'),
            (-1.8, 0, 'Not enough \nvariations in model', 'center', 'center'),
            (np.sqrt(0.5), np.sqrt(0.5), 'RMS ~ std(obs)', 'center', 'center'),
            (np.sqrt(0.5)/2, -np.sqrt(0.5)/2, 'RMS ~ 0.5.std(obs)', 'center', 'center')
        ]
        
        for x, y, s, ha, va in annotations:
            ax.annotate(s, (x, y), ha=ha, va=va, fontweight='bold')
    
    return fig