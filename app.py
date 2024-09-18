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

# Define a list of colors and line styles
colors = plt.cm.Set1(np.linspace(0, 1, 10))  # 10 distinct colors
line_styles = ['-', '--', ':', '-.']

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
            ui.input_select("variable", "Variable (main)", choices=[]),
            ui.input_select("variable_rates", "Variable (rates)", choices=[]),
            ui.input_action_button("load_button", "Load NetCDF data"),
        ),
        ui.output_plot("plot_output"),
        ui.output_plot("plot_output_rates"),
        ui.output_text("error_message"),
    ),
)

def server(input, output, session):
    datasets = reactive.Value({})
    datasets_rates = reactive.Value({})
    
    def get_list_from_input(input_string):
        return [item.strip() for item in input_string.split(',') if item.strip()]
    
    @reactive.Effect
    @reactive.event(input.load_button)
    def load_data():
        nonlocal datasets, datasets_rates
        new_datasets = {}
        new_datasets_rates = {}
        stations = get_list_from_input(input.stations())
        setups = get_list_from_input(input.setups())
        
        for setup in setups:
            for station in stations:
                dataset = load_netcdf(input.result_dir(), input.simulation_name(), setup, station)
                dataset_rates = load_netcdf(input.result_dir(), input.simulation_name(), setup, station, is_rates=True)
                if dataset is not None:
                    new_datasets[(setup, station)] = dataset
                if dataset_rates is not None:
                    new_datasets_rates[(setup, station)] = dataset_rates
        
        if new_datasets:
            datasets.set(new_datasets)
            variables = list(next(iter(new_datasets.values())).data_vars)
            ui.update_select("variable", choices=variables)
        else:
            datasets.set({})
            ui.update_select("variable", choices=[])
        
        if new_datasets_rates:
            datasets_rates.set(new_datasets_rates)
            variables_rates = list(next(iter(new_datasets_rates.values())).data_vars)
            ui.update_select("variable_rates", choices=variables_rates)
        else:
            datasets_rates.set({})
            ui.update_select("variable_rates", choices=[])

    @render.text
    def error_message():
        if not datasets() and not datasets_rates():
            return f"Error: Unable to load any NetCDF files. Please check the directory, simulation name, setups, and stations."
        return ""

    def plot_data(datasets, var, title):
        if not datasets:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        unique_stations = list(set(station for (setup, station) in datasets.keys()))
        unique_setups = list(set(setup for (setup, station) in datasets.keys()))
        
        color_cycle = cycle(colors)
        station_colors = {station: next(color_cycle) for station in unique_stations}
        
        line_style_cycle = cycle(line_styles)
        setup_line_styles = {setup: next(line_style_cycle) for setup in unique_setups}
        
        for (setup, station), ds in datasets.items():
            if var in ds:
                data = ds[var]
                label = f"{setup} - {station}"
                color = station_colors[station]
                linestyle = setup_line_styles[setup]
                
                if len(data.dims) == 1:
                    data.plot(ax=ax, label=label, color=color, linestyle=linestyle)
                elif len(data.dims) == 2:
                    data.isel({data.dims[1]: 0}).plot(ax=ax, label=label, color=color, linestyle=linestyle)
                else:
                    data.isel({dim: 0 for dim in data.dims[1:]}).plot(ax=ax, label=label, color=color, linestyle=linestyle)
        
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    @render.plot
    @reactive.event(input.variable, input.load_button)
    def plot_output():
        return plot_data(datasets(), input.variable(), input.variable())

    @render.plot
    @reactive.event(input.variable_rates, input.load_button)
    def plot_output_rates():
        return plot_data(datasets_rates(), input.variable_rates(), f"{input.variable_rates()} (Rates)")

app = App(app_ui, server)