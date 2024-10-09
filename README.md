Python Shiny app to visualize data from NetCDF generated within BCZ1D.

Tested with Positron, the new IDE of Posit.
Tested wit VSCode, with the Shiny extension.

# How to use the app

## Simulation 

1.  Specify the (full) path to your `simulation.yaml` (note: it can be named differently) and click on the button `Load YAML Configuration`. If the app finds the file, it will return a message saying that the YAML configuration was loaded successfully.
2.  The app will extract specific fields from the YAML file and will automatically display the result directory (should be a full path), the name of the simulation (note: do not change the latter two !), stations and setups (you can select one or several of them)
3.   Click on the `Load NetCDF data` button, wait for the data to load (should be fast enough).
4.   Select model variables from the list in *All Variables*.

## Validation 

### Validation tab

Validation plots where the observations are depicted in black dots and the model setup results are displayed as thick lines (monthly median of model outputs) with error bars (quartiles 0.25 and 0.75).

### Taylor diagram tab

The plot is automatically created based on the variable mapping specified in the `simulation.yaml` file. It only reacts to the number of selected setups.
