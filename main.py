# Copyright (C) 2020 Joris Nettelstroth

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""DataExplorer helps you explore correlations within your data.

It features a user interface that shows scatter plots of all the variables in
your data. Classes found in your data can be used to filter the views.

You can start this program with the "start_bokeh.cmd" batch file. This starts
a Bokeh server that can be accessed from any computer in the network.

This Python script 'main.py' is called if 'Bokeh serve' is used with the
folder name instead of a file name.

We create an initial set of example data and then create the
DataExplorer user interface.
"""

import argparse
from dataexplorer import Dataexplorer
from helpers import create_test_data, read_filetypes

# Read the user input:
description = 'Main module of the DataExplorer application. Needs to be '\
              'started by Bokeh and will not produce any output '\
              'on its own. You can use the available options to automatically'\
              ' load specify data. See the documentation for further help.'
parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)

parser.add_argument('-f', '--file', action='store', type=str,
                    dest='file_path',
                    help='Path to the file to load',
                    default=None)

parser.add_argument('-n', '--name', action='store', type=str,
                    dest='data_name',
                    help='Name of the loaded data',
                    default='Example Data')

parser.add_argument('--server_mode', action='store_true',
                    dest='server_mode',
                    help='Server mode enables the "leave page" confirmation '
                    'dialog and removes uploaded files after use',
                    default=False)

parser.add_argument('--bokeh_output_backend', action='store', type=str,
                    dest='output_backend',
                    help='Rendering options are "canvas", "webgl" and "svg"',
                    default='webgl')

args = parser.parse_args()

if args.file_path is not None:
    df = read_filetypes(args.file_path)
else:
    df = create_test_data()

DatEx = Dataexplorer(df, args.data_name, args.server_mode,
                     output_backend=args.output_backend)

# The script ends here (but Bokeh keeps waiting for user input)
