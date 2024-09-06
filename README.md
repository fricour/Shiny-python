# Shiny-python

Python Shiny app to visualize data from NetCDF generated within BCZ1D.

Tested with Positron, the new IDE of Posit.

# How to use

Specify the path to your BCZ1D directory, select your setup(s) and station(s) - you can also modify the `app.py` file to add more setups and/or stations - then click on the button `Load NetCDF data` to read the selected NetCDF files.
Once files are loaded, you can play with the variables and plot them. Note that the order of setups selection has its importance. If you have more variables in one setup (say you have more filtering organisms), you should select this one first.
