from shiny import App, render, ui, reactive
import xarray as xr
import matplotlib.pyplot as plt
import os
from itertools import cycle
import numpy as np
import yaml
import pandas as pd
import seaborn as sns

# Function to load the YAML configuration
def load_yaml_config(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None
    
# Function to load the Taylor data
def load_taylor_data(file_path):
    try:
        return pd.read_feather(file_path)
    except Exception as e:
        print(f"Error loading Taylor data: {e}")
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
    sns.scatterplot(data=df, x="NormalizedcRMSD", y='NormalizedBias', 
                    hue='variable', s=150, ax=ax, legend=False)
    
    # Add vertical and horizontal lines
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    
    # Add circular guidelines
    for radius in [0.5, 1]:
        circle = plt.Circle((0, 0), radius, color='b', fill=False, 
                            linestyle='--' if radius == 0.5 else '-')
        ax.add_patch(circle)
    
    # Add variable labels to each point
    for _, row in df.iterrows():
        ax.annotate(row['variable'], (row["NormalizedcRMSD"], row["NormalizedBias"]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Set plot limits
    #ax.set_xlim([-2, 2])
    #ax.set_ylim([-2, 2])
    
    # Add title and labels
    ax.set_title("Target Diagram", fontsize=16)
    ax.set_xlabel("Normalized cRMSD", fontsize=12)
    ax.set_ylabel("Normalized Bias", fontsize=12)
    
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

# Define a list of colors for setups
colors = plt.cm.Set1(np.linspace(0, 1, 10))  # 10 distinct colors

app_ui = ui.page_fluid(
    ui.include_css("styles.css"),
    ui.tags.div(
        ui.tags.h1("Investigate BCZ1D setups", class_="app-title"),
        class_="header"
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.tags.h3("Simulation"),
            ui.input_text("yaml_path", "Path to simulation.yaml", "/home/fricour/bcz1d/experiment/simulation.yaml"),
            ui.input_action_button("load_yaml", "Load YAML Configuration"),
            ui.output_text("yaml_load_status"),
            ui.input_text("result_dir", "Result directory"),
            ui.input_text("simulation_name", "Simulation name"),
            ui.input_selectize("stations", "Stations", choices=[], multiple=True),
            ui.input_selectize("setups", "Setups", choices=[], multiple=True),
            ui.input_action_button("load_button", "Load NetCDF data"),
            ui.input_selectize("variables", "All Variables", choices=[], multiple=True),
            ui.tags.h3("Validation"),            
            ui.input_selectize("yaml_variables", "Variables from YAML", choices=[], multiple=True),

        ),
        ui.navset_tab(
            ui.nav_panel("Simulation",
                ui.output_plot("plot_grid"),
                ui.output_text("error_message")
            ),
            ui.nav_panel("Validation",
                ui.output_plot("global_taylor_diagram")
            ),
        ),
    )
)

def server(input, output, session):
    config = reactive.Value(None)
    datasets = reactive.Value({})
    
    @reactive.Effect
    @reactive.event(input.load_yaml)
    def load_yaml_file():
        yaml_config = load_yaml_config(input.yaml_path())
        if yaml_config:
            config.set(yaml_config)
            ui.update_text("result_dir", value=yaml_config['paths']['result_dir'])
            ui.update_text("simulation_name", value=yaml_config['simulation_name'])
            ui.update_selectize("stations", choices=list(yaml_config['stations'].keys()))
            ui.update_selectize("setups", choices=yaml_config['setups'])
            ui.update_selectize("yaml_variables", choices=list(yaml_config['variable_mapping'].keys()))
        else:
            config.set(None)

    @render.text
    @reactive.event(input.load_yaml)
    def yaml_load_status():
        if config():
            return "YAML configuration loaded successfully."
        else:
            return "Failed to load YAML configuration. Please check the file path and try again."

    @reactive.Effect
    @reactive.event(input.load_button)
    def load_data():
        nonlocal datasets
        new_datasets = {}
        stations = input.stations()
        setups = input.setups()
        
        # Show loading notification
        with ui.Progress(min=0, max=len(stations) * len(setups)) as p:
            p.set(message="Loading NetCDF files", detail="Please wait...")
            
            for i, setup in enumerate(setups):
                for j, station in enumerate(stations):
                    dataset = load_netcdf(input.result_dir(), input.simulation_name(), setup, station)
                    dataset_rates = load_netcdf(input.result_dir(), input.simulation_name(), setup, station, is_rates=True)
                    if dataset is not None:
                        new_datasets[(setup, station, 'result')] = dataset
                    if dataset_rates is not None:
                        new_datasets[(setup, station, 'rates')] = dataset_rates
                    
                    # Update progress
                    p.set(value=i * len(stations) + j + 1)
        
        if new_datasets:
            datasets.set(new_datasets)
            variables = set()
            for ds in new_datasets.values():
                variables.update(ds.data_vars)
            ui.update_selectize("variables", choices=list(variables))
        else:
            datasets.set({})
            ui.update_selectize("variables", choices=[])

    @render.text
    def error_message():
        if not datasets():
            return f"Error: Unable to load any NetCDF files. Please check the directory, simulation name, setups, and stations."
        return ""

    @render.plot
    @reactive.event(input.variables, input.yaml_variables, input.load_button)
    def plot_grid():
        if not datasets() or (not input.variables() and not input.yaml_variables()):
            return None
        
        # Show loading notification for plot generation
        stations = input.stations()
        setups = input.setups()
        variables = input.variables()
            
        n_rows = len(variables)
        n_cols = len(stations)
            
        # Dynamic figure size calculation
        fig_width = min(24, max(12, 5 * n_cols))  # Min 12 inches, max 24 inches, 5 inches per column
        fig_height = min(100, max(8, 4 * n_rows))  # Min 8 inches, max 36 inches, 4 inches per row
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        #fig.suptitle(f"Simulation: {input.simulation_name()}", fontsize=16)
            
        setup_colors = {setup: color for setup, color in zip(setups, colors)}
            
        for row, var in enumerate(variables):
            for col, station in enumerate(stations):
                ax = axes[row, col]
                print(ax)
                    
                for setup in setups:
                    for data_type in ['result', 'rates']:
                        if (setup, station, data_type) in datasets():
                            ds = datasets()[(setup, station, data_type)]
                            if var in ds:
                                data = ds[var]
                                label = f"{setup} ({data_type})"
                                color = setup_colors[setup]
                                    
                                if len(data.dims) == 1:
                                    data.plot(ax=ax, label=label, color=color)
                                elif len(data.dims) == 2:
                                    data.isel({data.dims[1]: 0}).plot(ax=ax, label=label, color=color)
                                else:
                                    data.isel({dim: 0 for dim in data.dims[1:]}).plot(ax=ax, label=label, color=color)
                    
                #ax.set_title(f"{var} - {station}")
                ax.set_title(f"Station {station}")
                ax.legend(fontsize='small')
                ax.set_xlabel("Time")
                ax.set_ylabel(var)
            
        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        return fig
    
    @render.plot
    @reactive.event(input.yaml_variables)
    def global_taylor_diagram():
        if not input.yaml_variables() or not input.setups():
            return plt.figure()  # Return an empty figure if no variables are selected
        try:
            # Construct the correct path to the data_taylor.arrow file
            print(input.setups())
            setup = input.setups()[0]  # Only use the first setup # NOTE: This is a temporary solution
            file_path = os.path.join(input.result_dir(), input.simulation_name(), setup, "taylor_diagram_all_stations.arrow")
        
            # Read the feather file
            df = pd.read_feather(file_path)
        
            # Filter the dataframe based on selected variables
            selected_vars = input.yaml_variables()
            filtered_df = df[df['variable'].isin(selected_vars)]
        
            # Create the target diagram
            fig = create_target_diagram(filtered_df, add_annotations=True)
        
            return fig
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return plt.figure(figsize=(10, 10))  # Return an empty figure
        except Exception as e:
            print(f"Error creating Taylor diagram: {e}")
            return plt.figure(figsize=(10, 10))  # Return an empty figure

app = App(app_ui, server)