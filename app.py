from shiny import App, render, ui, reactive
import xarray as xr
import matplotlib.pyplot as plt
import os
from itertools import cycle
import numpy as np

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
            ui.input_text("result_dir", "Result directory", "/home/fricour/bcz1d/results"),
            ui.input_text("simulation_name", "Simulation name", ""),
            ui.input_text("stations", "Stations (comma-separated)", ""),
            ui.input_text("setups", "Setups (comma-separated)", ""),
            ui.input_selectize("variables", "Variables", choices=[], multiple=True),
            ui.input_action_button("load_button", "Load NetCDF data"),
        ),
        ui.output_plot("plot_grid"),
        ui.output_text("error_message"),
    ),
)

def server(input, output, session):
    datasets = reactive.Value({})
    
    def get_list_from_input(input_string):
        return [item.strip() for item in input_string.split(',') if item.strip()]
    
    @reactive.Effect
    @reactive.event(input.load_button)
    def load_data():
        nonlocal datasets
        new_datasets = {}
        stations = get_list_from_input(input.stations())
        setups = get_list_from_input(input.setups())
        
        for setup in setups:
            for station in stations:
                dataset = load_netcdf(input.result_dir(), input.simulation_name(), setup, station)
                dataset_rates = load_netcdf(input.result_dir(), input.simulation_name(), setup, station, is_rates=True)
                if dataset is not None:
                    new_datasets[(setup, station, 'result')] = dataset
                if dataset_rates is not None:
                    new_datasets[(setup, station, 'rates')] = dataset_rates
        
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
        
        stations = get_list_from_input(input.stations())
        setups = get_list_from_input(input.setups())
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