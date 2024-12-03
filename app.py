from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import os
from itertools import cycle
import numpy as np
import pandas as pd
from ratelimit import debounce
from ipyleaflet import Map, Marker, AwesomeIcon
from shinywidgets import output_widget, render_widget
import custom_functions as cf

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
            ui.tags.h3("Simulation", style="font-weight: bold;"),
            ui.input_text("yaml_path", "Path to simulation.yaml", "/home/fricour/bcz1d/experiment/simulation_CHL.yaml"),
            #ui.input_text("yaml_path", "Path to simulation.yaml", "/home/fricour/bcz1d/experiment/test.yaml"),
            #ui.input_text("yaml_path", "Path to simulation.yaml", "/home/arthur/Desktop/DOCS/PROJECTS/bcz1d_DEV/bcz1d/experiment/simulation_CHL_GE.yaml"),
            ui.input_action_button("load_yaml", "Load YAML Configuration"),
            ui.output_text("yaml_load_status"),
            ui.input_text("result_dir", "Result directory"),
            ui.input_text("simulation_name", "Simulation name"),
            ui.input_selectize("stations", "Stations", choices=[], multiple=True),
            ui.input_selectize("setups", "Setups", choices=[], multiple=True),
            ui.input_action_button("load_button", "Load NetCDF data"),
            ui.input_selectize("variables", "All Variables", choices=[], multiple=True),
            ui.tags.hr(style="border-top: 2px solid #ccc; margin-top: 20px; margin-bottom: 20px;"),
            ui.tags.h3("Validation", style="font-weight: bold;"),            
            ui.input_selectize("yaml_variables", "Variables from YAML", choices=[], multiple=True)
        ),
        ui.navset_tab(
            ui.nav_panel("Time Series",
                ui.output_ui("dynamic_card_plot_grid"),
            ),
            ui.nav_panel("Vertical Profiles",
                ui.output_ui("dynamic_card_plot_vertical"), 
            ),
            ui.nav_panel("Validation",
                ui.output_ui("dynamic_card_plot_validation"),
            ),
            ui.nav_panel("Target Diagram",
                ui.card(
                    ui.output_plot("plot_target"),
                    height="calc(100vh - 100px)" # Adjust the 100px value as needed to account for the header
                )
            ),
            ui.nav_panel("Map",
                ui.card(
                    output_widget("map"),
                    height="calc(100vh - 100px)" # Adjust the 100px value as needed to account for the header
                )
            ),
        ),
        ui.tags.footer(
            ui.tags.div(
                "Made with ❤️ by Florian Ricour",
                class_="footer-content"
            ),
            class_="footer"
        )
    )
)

def server(input, output, session):
    config = reactive.Value(None)
    datasets = reactive.Value({})
    
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
            #ui.update_selectize("yaml_variables", choices=list(yaml_config['variable_mapping'].keys()), selected=list(yaml_config['variable_mapping'].keys()))
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

    # DYNAMIC CARD PLOT FOR GRID PLOT   
    @render.ui
    def dynamic_card_plot_grid():
        # Calculate card height based on number of subplots
        card_height = f"{400 * (len(input.variables()))}px"
        return ui.card(
            ui.output_plot("plot_grid"),
            ui.output_text("error_message"),
            style=f"height: {card_height};"
        )
        
    # TIME SERIES GRID PLOT
    @render.plot
    @debounce(1) # Debounce (1 second) the plot generation to avoid multiple calls (when selecting the input variables)
    @reactive.event(input.variables, input.load_button, input.setups, input.stations)
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
    
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex='col', sharey='row',
                                 figsize=(fig_width, fig_height), constrained_layout=True)
            
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
                    # ax.set_xticks([pd.Timestamp(f"2020-{month:02d}-01") for month in [1, 4, 7, 10]]) # without this, it is not working
                    # ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
                    ax.set_xlabel("Month")
                if col != 0 :
                    ax.set_yticklabels([])
                # Long name to be used only if there's enough room    
                # ax.set_ylabel(var_long_name if col==0 else "" )
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
    def error_message_vertical(): 
        if not datasets():
            return f"Error: Unable to load any NetCDF files. Please check the directory, simulation name, setups, and stations."
        return ""

    # DYNAMIC CARD PLOT FOR VERTICAL PROFILES PLOT
    @render.ui
    def dynamic_card_plot_vertical():
        # Calculate card height based on number of subplots
        card_height = f"{400 * (len(input.variables()))}px"
        return ui.card(
            ui.output_plot("plot_vertical"),
            ui.output_text("error_message_vertical"),
            style=f"height: {card_height};"
        )

    # VERTICAL PROFILES PLOT
    @render.plot
    @debounce(1) # Debounce (1 second) the plot generation to avoid multiple calls (when selecting the input variables)
    @reactive.event(input.variables, input.load_button, input.setups, input.stations)
    def plot_vertical():
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

                ymin,ymax = ax.get_ylim()
                ymins.append(ymin)
                ymaxs.append(ymax)
            
            for col, station in enumerate(stations):
                ax = axes[row, col]
                ax.set_ylim(np.asarray(ymins).min(), np.asarray(ymaxs).max())

        return fig
    

    # TARGET DIAGRAM PLOT
    @render.plot
    def plot_target():
        yaml_config = cf.load_yaml_config(input.yaml_path())
        try:
            dfs = []
            for i in range(len(input.setups())):
                setup = input.setups()[i]
                file_path = os.path.join(input.result_dir(), input.simulation_name(), setup, "taylor_diagram_all_stations.arrow")
        
                # Read the feather file
                df = pd.read_feather(file_path)
                dfs.append(df)

            # Concatenate the dataframes
            df = pd.concat(dfs)

            # Filter the dataframe based on selected variables
            selected_vars = list(yaml_config['variable_mapping'].keys())
            filtered_df = df[df['variable'].isin(selected_vars)]
        
            # Create the target diagram
            fig = cf.create_target_diagram(filtered_df, add_annotations=False)
        
            return fig
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return plt.figure(figsize=(10, 10))  # Return an empty figure
        except Exception as e:
            print(f"Error creating Taylor diagram: {e}")
            return plt.figure(figsize=(10, 10))  # Return an empty figure
        
    @render.ui
    def dynamic_card_plot_validation():
        # Calculate card height based on number of subplots
        card_height = f"{400 * (len(input.yaml_variables()))}px"
        return ui.card(
            ui.output_plot("plot_validation"),
            style=f"height: {card_height};"
        )
    
    # VALIDATION PLOT
    @render.plot
    @debounce(0.1)
    @reactive.event(input.yaml_variables, input.setups, input.stations)
    def plot_validation():
        if not input.yaml_variables():
            return plt.figure()
        try:
            dfs = []
            stats_dfs = []
            for i in range(len(input.setups())):
                setup = input.setups()[i]
                df_path = os.path.join(input.result_dir(), input.simulation_name(), setup, "validation_data.arrow")
                stats_path = os.path.join(input.result_dir(), input.simulation_name(), setup, "validation_monthly_stats.arrow")
        
                # Read the feather file
                df = pd.read_feather(df_path)
                dfs.append(df)
                stats_df = pd.read_feather(stats_path)
                stats_dfs.append(stats_df)

            # Concatenate the dataframes
            df = pd.concat(dfs)
            stats_df = pd.concat(stats_dfs)

            # Filter the dataframe based on selected variables
            selected_vars = input.yaml_variables()
            filtered_df = df[df['variable'].isin(selected_vars)]
            filtered_stats_df = stats_df[stats_df['variable'].isin(selected_vars)]

            # read input data
            stations = input.stations()
            variables = input.yaml_variables()
            setups = input.setups()
            
            n_rows = len(variables)
            n_cols = len(stations)
            
            # Dynamic figure size calculation
            fig_width = min(24, max(12, 5 * n_cols))  # Min 12 inches, max 24 inches, 5 inches per column
            fig_height = min(100, max(8, 4 * n_rows))  # Min 8 inches, max 36 inches, 4 inches per row
    
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
            
            setup_colors = {setup: color for setup, color in zip(setups, colors)}
            
            for row, var in enumerate(variables): 
                ymins, ymaxs = [], []
                for col, station in enumerate(stations): 
                    ax = axes[row, col]
                    for k, setup_name in enumerate(np.unique(filtered_df['setup'])):
                        #station_data = filtered_df[(filtered_df['variable'] == var) & (filtered_df['station'] == station) & (filtered_df['setup'] == setup_name)]
                        station_stats = filtered_stats_df[(filtered_stats_df['variable'] == var) & (filtered_stats_df['station'] == station) & (filtered_stats_df['setup'] == setup_name)]

                        months = np.arange(1, 13)

                        # Choose a different color for each setup
                        #setup_color = sns.color_palette()[k]

                        # Plot individual observations and boxplot
                        #for month in months:
                        #    # add boxplot
                        #    bp = station_data.boxplot(column='obs', by='month', ax=axs[i,j],
                        #                        grid = False,
                        #                        notch=False, # rectangle boxplot if False
                        #                        showfliers=True, # no outliers, we already plot the observations
                        #                        medianprops={"color": "black", "linewidth": 2},
                        #                        showcaps=False, # no caps
                        #                        boxprops={"color": "black", "linewidth": 2},
                        #                        whiskerprops={"color": "black", "linewidth": 0}
                        #                        )
                            # add individual observations
                        #    monthly_data = station_data[station_data['month'] == month]
                        #    axs[i,j].scatter([month] * len(monthly_data), monthly_data['obs'], alpha=0.5, color="black", s=10)

                        # Add the median observation values (only one black line in legend)
                        if k == 0:
                            ax.plot(months, station_stats['obs-median'], color='black', label=f"Observations", linewidth=2)
                        else:
                            ax.plot(months, station_stats['obs-median'], color='black', linewidth=2)
                        
                        # Add error bars
                        ax.errorbar(months, station_stats['obs-median'], 
                                    yerr=[station_stats['obs-median'] - station_stats['obs-q25'], 
                                          station_stats['obs-q75'] - station_stats['obs-median']],
                                    fmt='none', color='black', ecolor='black', capsize=5, alpha=1, elinewidth=2, capthick=2)

                        # Add the median model values
                        ax.plot(months, station_stats['mod-median'], color=setup_colors[setup_name], label=f"{setup_name}", linewidth=2)

                        # Add error bars
                        ax.errorbar(months, station_stats['mod-median'], 
                                    yerr=[station_stats['mod-median'] - station_stats['mod-q25'], 
                                          station_stats['mod-q75'] - station_stats['mod-median']],
                                    fmt='none', color=setup_colors[setup_name], ecolor=setup_colors[setup_name], capsize=5, alpha=1, elinewidth=2, capthick=2)

                        ax.set_title(f"{station}" if row==0 else "" )
                        if row != (len(variables)-1):
                            ax.tick_params(axis='x',labelbottom='off')
                            ax.set_xticklabels([''])
                        else:
                            # Add month labels for every three months
                            ax.set_xticks([1, 4, 7, 10])
                            ax.set_xticklabels(['Jan', 'Apr', 'Jul', 'Oct'])
                            ax.set_xlabel("Month")
                        #if col != 0: 
                        #    ax.set_yticklabels([])
                        ax.set_ylabel(var if col==0 else "" )
                        ax.grid(True)

                        if (row==0) and (col==0):
                            ax.legend(fontsize='small')

                        ymin,ymax = ax.get_ylim()
                        ymins.append(ymin)
                        ymaxs.append(ymax)
            
                for col, station in enumerate(stations):
                    ax.set_ylim(np.asarray(ymins).min(), np.asarray(ymaxs).max())

        except FileNotFoundError:
            print(f"File not found: {df_path} or {stats_path}")
            return plt.figure(figsize=(10, 10)) # Return an empty figure
        except Exception as e:
            print(f"Error creating validation plots: {e}")
            return plt.figure(figsize=(10, 10))
        
        return fig
    
    # MAP PLOT
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
