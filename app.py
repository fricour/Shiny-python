from shiny import App, render, ui, reactive
import xarray as xr
import matplotlib.pyplot as plt
import os
from itertools import cycle
import numpy as np
import yaml

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
            ui.input_text("yaml_path", "Path to simulation.yaml", "simulation.yaml"),
            ui.input_action_button("load_yaml", "Load YAML Configuration"),
            ui.output_text("yaml_load_status"),
            ui.input_text("result_dir", "Result directory"),
            ui.input_text("simulation_name", "Simulation name"),
            ui.input_selectize("stations", "Stations", choices=[], multiple=True),
            ui.input_selectize("setups", "Setups", choices=[], multiple=True),
            ui.input_selectize("variables", "Variables", choices=[], multiple=True),
            ui.input_action_button("load_button", "Load NetCDF data"),
        ),
        ui.output_plot("plot_grid"),
        ui.output_text("error_message"),
    ),
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
    @reactive.event(input.variables, input.load_button)
    def plot_grid():
        if not datasets() or not input.variables():
            return None
        
        # Show loading notification for plot generation
        stations = input.stations()
        setups = input.setups()
        variables = input.variables()
            
        n_rows = len(variables)
        n_cols = len(stations)
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        fig.suptitle(f"Simulation: {input.simulation_name()}", fontsize=16)
            
        setup_colors = {setup: color for setup, color in zip(setups, colors)}
            
        for row, var in enumerate(variables):
            for col, station in enumerate(stations):
                ax = axes[row, col]
                    
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
                    
                ax.set_title(f"{var} - {station}")
                ax.legend(fontsize='small')
                    
                if row == n_rows - 1:
                    ax.set_xlabel("Time")
                if col == 0:
                    ax.set_ylabel(var)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        return fig

app = App(app_ui, server)