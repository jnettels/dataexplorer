# -*- coding: utf-8 -*-

'''
The DataExplorer should help you explore correlations within your data. It
features a user interface that shows scatter plots of all the variables in
your data. Classes found in your data can be used to filter the views.

You can start this program with the "start_bokeh.cmd" batch file. This starts
a Bokeh server that can be accessed from any computer in the network.

This Python script 'main.py' is called if 'Bokeh serve' is used with the
folder name instead of a file name.

We create an initial set of example data and then create the
DataExplorer user interface.
'''

from helpers import create_test_data
from dataexplorer import Dataexplorer

df = create_test_data()
data_name = 'Example Data'

DatEx = Dataexplorer(df, data_name)

# The script ends here (but Bokeh keeps waiting for user input)
