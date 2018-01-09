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

from optparse import OptionParser  # For parsing options with the program call
from dataexplorer import Dataexplorer
from helpers import create_test_data, read_filetypes

# Read the user input:
parser = OptionParser()
parser.add_option('-f', '--file', action='store', type='string',
                  dest='file_path',
                  help='Path to the file to load.',
                  default=None)

parser.add_option('-n', '--name', action='store', type='string',
                  dest='data_name',
                  help='Name of the loaded data; default = %default',
                  default='Example Data')

options, args = parser.parse_args()

if options.file_path is not None:
    try:
        df = read_filetypes(options.file_path)
    except Exception:
        raise  # Reraise the current exception
else:
    df = create_test_data()

DatEx = Dataexplorer(df, options.data_name)

# The script ends here (but Bokeh keeps waiting for user input)
