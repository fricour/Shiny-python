from shiny import App, render, ui, reactive
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
import yaml
import pandas as pd
import seaborn as sns
from ratelimit import debounce
from ipyleaflet import Map, Marker, AwesomeIcon
from shinywidgets import output_widget, render_widget
import custom_functions as cf
from matplotlib.colors import LogNorm
import cmocean
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define a list of colors for setups
colors = plt.cm.Set1(np.linspace(0, 1, 10))  # 10 distinct colors

app_ui = ui.page_fluid(
    ui.include_css("styles.css"),
    ui.tags.div(
        ui.tags.h1("Argo1D", class_="app-title"),
        class_="header"
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.tags.h3("Observations", style="font-weight: bold;"),   
            ui.input_selectize("argo_variables", "Argo variables", choices=["CHLA", "BBP700", "DOXY", "PSAL", "TEMP"], selected=["CHLA"], multiple=False),
            ui.tags.hr(style="border-top: 2px solid #ccc; margin-top: 20px; margin-bottom: 20px;"),   
            ui.tags.h3("Simulation", style="font-weight: bold;"),
            ui.input_text("yaml_path", "Path to simulation.yaml", "/home/fricour/bcz1d/experiment/simulation_CHL.yaml"),
            ui.input_action_button("load_yaml", "Load YAML Configuration"),
            ui.output_text("yaml_load_status"),
            ui.input_text("result_dir", "Result directory"),
            ui.input_text("simulation_name", "Simulation name"),
            ui.input_selectize("stations", "Stations", choices=[], multiple=True),
            ui.input_selectize("setups", "Setups", choices=[], multiple=True),
            ui.input_action_button("load_button", "Load NetCDF data"),
            ui.input_selectize("variables", "All Variables", choices=[], multiple=True),
            ui.input_action_button("refresh_plot", "Show plots"),
        ),
        ui.navset_tab(
            ui.nav_panel("BGC-ARGO vs Model",
                ui.layout_columns(
                    ui.card(
                        ui.output_plot("plot_trajectory"),
                        full_screen=False
                    ),
                    col_widths=[6],  # Full width for trajectory
                    height="px"  # Set explicit height for trajectory
                ),
                ui.layout_columns(
                    ui.card(
                        ui.output_plot("plot_argo_data"),
                        full_screen=False
                    ),
                    ui.card(
                        ui.output_plot("plot_model_data"),
                        ui.output_text("error_message2"),
                        full_screen=False
                    ),
                    col_widths=[6, 6],  # Equal width for bottom two plots
                    height="calc(100vh - 550px)"
                )
            ),
            ui.nav_panel("Time Series",
                ui.card(
                    ui.output_plot("plot_grid"),
                    ui.output_text("error_message"),
                    height="calc(150vh - 100px)" # Adjust the 100px value as needed to account for the header
                ),
            ),
            ui.nav_panel("Indicators",
                ui.card(
                    ui.output_plot("plot_vert"),
                    ui.output_text("error_message_vert"),
                    height="calc(150vh - 100px)" 
                ),
            ),
        ),
        ui.tags.footer(
            ui.tags.div(
                "Made with ❤️ by Team PinkFish",
                class_="footer-content"
            ),
            class_="footer"
        )
    )
)

def server(input, output, session):
    config = reactive.Value(None)
    datasets = reactive.Value({})

    # Define variable properties
    VAR_PROPERTIES = {
        "CHLA": {
            "unit": "mg/m³",
            "cmap": cmocean.cm.algae,
            "log_scale": True,
            "model_var": "total_chlorophyll"
        },
        "BBP700": {
            "unit": "m⁻¹",
            "cmap": cmocean.cm.turbid,
            "log_scale": True,
            "model_var": "bbp_700"
        },
        "DOXY": {
            "unit": "μmol/kg",
            "cmap": cmocean.cm.oxy,
            "log_scale": False,
            "model_var": "oxy_O2"
        },
        "TEMP": {
            "unit": "°C",
            "cmap": cmocean.cm.thermal,
            "log_scale": False,
            "model_var": "temp"
        },
        "PSAL": {
            "unit": "PSU",
            "cmap": cmocean.cm.haline,
            "log_scale": False,
            "model_var": "salt"
        }
    }

    # non reactive data for the moment
    argo_data = xr.open_dataset("./data/1902381_Sprof.nc")

    # results
    simulation1d_data = xr.open_dataset("./data/1902381_result.nc")

    @render.plot
    def plot_trajectory():
        # Create figure with Cartopy projection
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        
        # Calculate extent with larger buffer to show nearby land
        lon_min, lon_max = argo_data.LONGITUDE.min(), argo_data.LONGITUDE.max()
        lat_min, lat_max = argo_data.LATITUDE.min(), argo_data.LATITUDE.max()
        
        # Expand extent significantly to show surrounding land
        lon_buffer = max(5, (lon_max - lon_min) * 1.5)  # At least 10 degrees buffer
        lat_buffer = max(5, (lat_max - lat_min) * 1.5)
        
        extent = [
            lon_min - lon_buffer,
            lon_max + lon_buffer,
            lat_min - lat_buffer,
            lat_max + lat_buffer
        ]
        
        # Set map extent first
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Add map features with better visibility
        ax.add_feature(cfeature.LAND, facecolor='tan', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', alpha=0.5)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False, 
                        linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot trajectory
        scatter = ax.scatter(
            argo_data.LONGITUDE, argo_data.LATITUDE, 
            cmap='viridis',
            s=50,
            transform=ccrs.PlateCarree(),
            zorder=5,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Mark start and end points
        ax.plot(argo_data.LONGITUDE.values[0], argo_data.LATITUDE.values[0], 
                'go', markersize=12, label='Start', 
                transform=ccrs.PlateCarree(), zorder=6, markeredgecolor='black')
        ax.plot(argo_data.LONGITUDE.values[-1], argo_data.LATITUDE.values[-1], 
                'rs', markersize=12, label='End', 
                transform=ccrs.PlateCarree(), zorder=6, markeredgecolor='black')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        return fig


    @render.plot
    def plot_model_data():
        var = input.argo_variables()
        ds = simulation1d_data
        
        # Get variable properties
        var_props = VAR_PROPERTIES.get(var, {
            "unit": "",
            "cmap": "viridis",
            "log_scale": False,
            "model_var": ""
        })
        
        # Transpose the dataset
        ds_transposed = ds.transpose()
        
        # Create figure with better size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get the data variable 
        data_var = ds_transposed[var_props['model_var']] 
        
        # Plot with transposed data
        plot_kwargs = {
            'ax': ax,
            'cmap': var_props['cmap'],
            'cbar_kwargs': {
                'label': f'{var} [{var_props["unit"]}]',
                'shrink': 0.8,
                'pad': 0.02
            }
        }
        
        # Add log scale if specified
        if var_props['log_scale']:
            import matplotlib.colors as colors
            plot_kwargs['norm'] = colors.LogNorm()
        
        data_var.plot(**plot_kwargs)
        
        # Improve axis labels
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Depth [m]', fontsize=12)
        ax.set_title('')
        
        # Invert y-axis so depth increases downward
        #ax.invert_yaxis()
        
        # Improve grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Tighter layout
        plt.tight_layout()
        
        return fig

    @render.plot
    def plot_argo_data():
        var = input.argo_variables()
        ds = argo_data
        
        # Extract and prepare time/pressure data (from prepare_common_data)
        time = ds['JULD'].values
        time = np.repeat(time, ds['PRES'].shape[1])
        time = pd.to_datetime(time)
        
        pres = ds['PRES'].values.flatten()
        
        base_mask = ~np.isnan(pres) #& (pres <= 600)
        
        # Extract variable data
        var_data = ds[var].values.flatten()
        mask = base_mask & ~np.isnan(var_data)
        
        var_time = time[mask]
        var_pres = pres[mask]
        var_values = var_data[mask]
        
        # Check for adjusted data
        if f"{var}_ADJUSTED" in ds:
            var_data_adj = ds[f"{var}_ADJUSTED"].values.flatten()
            var_data_adj = var_data_adj[mask]
             
            if not np.isnan(var_data_adj).all():
                var_values = var_data_adj
                # Filter any NaN values in adjusted data
                adj_mask = ~np.isnan(var_data_adj)
                var_time = var_time[adj_mask]
                var_pres = var_pres[adj_mask]
                var_values = var_values[adj_mask]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))


        # Get properties for current variable
        var_name = input.argo_variables()
        props = VAR_PROPERTIES.get(var_name, {
            "unit": "",
            "cmap": "viridis",
            "log_scale": False
        })

        if props["log_scale"]:
            # Filter out non-positive values for log scale
            positive_mask = var_values > 0
            plot_time = var_time[positive_mask]
            plot_pres = var_pres[positive_mask]
            plot_values = var_values[positive_mask]
            sc = ax.scatter(plot_time, plot_pres, c=plot_values, cmap=props["cmap"], s=20,
                    norm=LogNorm(vmin=plot_values.min(), vmax=plot_values.max()))
        else:
            sc = ax.scatter(var_time, var_pres, c=var_values, cmap=props["cmap"], s=20)

        ax.invert_yaxis()
        ax.set_ylim(600, 0)  
        ax.set_xlabel('Time')
        ax.set_ylabel('Pressure (dbar)')

        cbar = plt.colorbar(sc, ax=ax)
        if props["log_scale"]:
            cbar.set_label(f'{var_name} ({props["unit"]}, log scale)')
        else:
            cbar.set_label(f'{var_name} ({props["unit"]})')

        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    @reactive.Effect
    @reactive.event(input.load_yaml)
    def load_yaml_file():
        yaml_config = cf.load_yaml_config(input.yaml_path())
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
        try:
            new_datasets = {}
            stations = input.stations()
            setups = input.setups()
        
            # Show loading notification
            with ui.Progress(min=0, max=len(stations) * len(setups)) as p:
                p.set(message="Loading NetCDF files", detail="Please wait...")
            
                for i, setup in enumerate(setups):
                    for j, station in enumerate(stations):
                        try:
                            dataset = cf.load_netcdf(input.result_dir(), input.simulation_name(), setup, station)
                            dataset_rates = cf.load_netcdf(input.result_dir(), input.simulation_name(), setup, station, is_rates=True)
                            if dataset is not None:
                                new_datasets[(setup, station, 'result')] = dataset
                            if dataset_rates is not None:
                                new_datasets[(setup, station, 'rates')] = dataset_rates
                        except Exception as e:
                            print(f"Error loading data for setup {setup}, station {station}: {str(e)}")
                    
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
        except Exception as e:
            print(f"Error in load_data: {str(e)}")
            ui.notification_show(f"Error loading data: {str(e)}", type="error")

    @render.text
    def error_message(): 
        if not datasets():
            return f"Error: Unable to load any NetCDF files. Please check the directory, simulation name, setups, and stations."
        return ""

    @render.text
    def error_message2(): 
        if simulation1d_data.total_chlorophyll.values.shape[0]==0:
            return f"Did you launch the simulation?"
        return ""

    @render.plot
    #@debounce(1) # Debounce (1 second) the plot generation to avoid multiple calls (when selecting the input variables)
    #@reactive.event(input.variables, input.load_button, input.setups, input.stations)
    @reactive.event(input.refresh_plot)
    def plot_grid():
        if not datasets() or (not input.variables()):
            return None

        # Show loading notification for plot generation
        stations = input.stations()
        setups = input.setups()
        variables = input.variables()
            
        n_rows = len(variables)
        n_cols = len(stations)
            
        # Dynamic figure size calculation
        fig_width = min(24, max(12, 5 * n_cols))  # Min 12 inches, max 24 inches, 5 inches per column
        fig_height = min(100, max(8, 6 * n_rows))  # Min 8 inches, max 36 inches, 4 inches per row
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
            
        setup_colors = {setup: color for setup, color in zip(setups, colors)}
            
        for row, var in enumerate(variables):
            
            ymins, ymaxs = [], []            
            for col, station in enumerate(stations):
                ax = axes[row, col]
                    
                for setup in setups:
                    for data_type in ['result', 'rates']:
                        if (setup, station, data_type) in datasets():
                            ds = datasets()[(setup, station, data_type)]
                            if var in ds:
                                #print(ds)
                                data = ds[var].squeeze() # !! Without the squeeze, data retain a lon and lat dimension, so ndims ==4, and only the later case below applies. 
                                var_long_name = data.attrs.get('long_name')
                                #print(var_long_name)
                                label = f"{setup}"
                                color = setup_colors[setup]
                                if len(data.dims) == 1: # -> Assumed : The only dimension is time. 
                                    data.plot(ax=ax, label=label, color=color)
                                elif len(data.dims) == 2: # -> Assumed : The two dimensions are time and depth.
                                    # TODO : include a controller to choose between surface, bottom, or vertical average values.
                                    # TODO :  ... at somoe point, vertical integral can also be an option (! consider dz, maybe dynamically)
                                    # TODO :  ... it could be relevant, to have this choice specifically available for all variables .. 
                                    # data.isel({data.dims[1]: 0}).plot(ax=ax, label=label, color=color)
                                    data.mean(data.dims[1]).plot(ax=ax, label=label, color=color)
                                else:  # -> I don't know when this case i supposed to apply,...
                                    data.isel({dim: 0 for dim in data.dims[1:]}).plot(ax=ax, label=label, color=color)
                    
                ax.set_title(f"{station}" if row==0 else "" )
                if row != (len(variables) - 1):
                # Remove x-axis ticks and labels for all rows except the last
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    ax.set_xlabel('')
                else: # keep this else statement for later if needed
                    # For the last row, set only 4 month ticks
                    #ax.set_xticks([pd.Timestamp(f"2020-{month:02d}-01") for month in [1, 4, 7, 10]]) # without this, it is not working
                    #ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
                    ax.set_xlabel("Month")
                if col != 0 :
                    ax.set_yticklabels([])
                # Long name to be used only if there's enough room    
#                ax.set_ylabel(var_long_name if col==0 else "" )
                ax.set_ylabel('%s \n [%s]'%(var.split('_')[-1], data.attrs.get('units') ) if col==0 else "" )
                ax.grid()

                if (row==0) and (col==0):
                    ax.legend(fontsize='small')

                ymin,ymax = ax.get_ylim()
                ymins.append(ymin)
                ymaxs.append(ymax)
            
            for col, station in enumerate(stations):
                ax = axes[row, col]
                ax.set_ylim(np.asarray(ymins).min(), np.asarray(ymaxs).max())

        return fig


    @render.text
    def error_message_vert(): 
        if not datasets():
            return f"Error: Unable to load any NetCDF files. Please check the directory, simulation name, setups, and stations."
        return ""

    @render.plot
    #@debounce(1) # Debounce (1 second) the plot generation to avoid multiple calls (when selecting the input variables)
    #@reactive.event(input.variables, input.load_button, input.setups, input.stations)
    @reactive.event(input.refresh_plot)
    def plot_vert():
        if not datasets() or (not input.variables()):
            return None

        # Show loading notification for plot generation
        stations = input.stations()
        setups = input.setups()
        variables = input.variables()
            
        n_rows = len(variables)
        n_cols = len(stations)
            
        # Dynamic figure size calculation
        fig_width = min(24, max(12, 5 * n_cols))  # Min 12 inches, max 24 inches, 5 inches per column
        fig_height = min(100, max(8, 6 * n_rows))  # Min 8 inches, max 36 inches, 4 inches per row
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
            
        setup_colors = {setup: color for setup, color in zip(setups, colors)}
            
        for row, var in enumerate(variables):
            
            ymins, ymaxs = [], []            
            for col, station in enumerate(stations):
                ax = axes[row, col]
                    
                # Showing only the first setup
                for setup in [setups[0]]: 
                    for data_type in ['result', 'rates']:
                        if (setup, station, data_type) in datasets():
                            ds = datasets()[(setup, station, data_type)]
                            if var in ds:
                                data = ds[var].squeeze() # !! Without the squeeze, data retain a lon and lat dimension, so ndims ==4, and only the later case below applies. 
                                var_long_name = data.attrs.get('long_name')
                                label = f"{setup} "
                                color = setup_colors[setup]
                                if len(data.dims) == 1: # -> Assumed : The only dimension is time. 
                                    data.plot(ax=ax, label=label, color=color)
                                elif len(data.dims) == 2: # -> Assumed : The two dimensions are time and depth.
                                    data.plot.contourf(ax=ax, x='time', levels =25, cmap = 'viridis', label=label)
                                else:  # -> I don't know when this case applies,...
                                    data.isel({dim: 0 for dim in data.dims[1:]}).plot(ax=ax, label=label, color=color)
                    
                ax.set_title(f"{station}" if row==0 else "" )
                ax.grid()

                # remove y-label for all columns except the first one
                if col != 0:
                    ax.set_ylabel("")

                ymin,ymax = ax.get_ylim()
                ymins.append(ymin)
                ymaxs.append(ymax)
            
            for col, station in enumerate(stations):
                ax = axes[row, col]
                ax.set_ylim(np.asarray(ymins).min(), np.asarray(ymaxs).max())

        return fig

    
    @output
    @render_widget
    def map():

        # create fun icon
        custom_icon = AwesomeIcon(
                name='life-ring',
                marker_color='white',
                icon_color='red',
                spin=True
            )

        map = Map(center=(52, 3), zoom=7)
        stations = input.stations()
        for station in stations:

            # station position
            station_pos = (config()['stations'][station]['lat'], config()['stations'][station]['lon'])
            
            # add marker
            point = Marker(location=station_pos,
                           radius=5,
                           icon=custom_icon,
                           draggable=False)
            
            map.add_layer(point)
        return map

app = App(app_ui, server)