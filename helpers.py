# -*- coding: utf-8 -*-
'''
Created on Fri Nov 24 09:54:53 2017

@author: nettelstroth

Additional helper functions for the script dataexplorer.py

'''

import os
import logging
import base64
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, CustomJS, HoverTool
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
from bokeh.models.widgets import Button
from bokeh import palettes
from math import pi

from bokeh.plotting import figure


def create_test_data():
    '''Create some test data. Includes sin, cos and some random numbers. The
    amount of data is hardcoded and controlled with the number of time steps.

    Args:
        None

    Returns:
        df (Pandas DataFrame) : An example set of test data.
    '''

    time_steps = 100  # Control the amount of test data

    new_index = pd.date_range(start=pd.to_datetime('today'),
                              periods=time_steps, freq='D')

    dataA1 = {'T1': np.random.randint(0, 20, time_steps),
              'T2': np.random.randint(0, 30, time_steps),
              'Sin': np.sin(np.linspace(-np.pi, np.pi, time_steps)),
              'Cos': np.cos(np.linspace(-np.pi, np.pi, time_steps)),
              'Category Label 1': pd.Categorical(['A']*time_steps),
              'Category Label 2': pd.Categorical(['First']*time_steps),
              'Category Label 3': pd.Categorical(['10-20']*time_steps),
              }
    dataA2 = {'T1': np.random.randint(10, 30, time_steps),
              'T2': np.random.randint(10, 40, time_steps),
              'Category Label 1': pd.Categorical(['A']*time_steps),
              'Category Label 2': pd.Categorical(['Second']*time_steps),
              'Category Label 3': pd.Categorical(['10-20']*time_steps),
              }
    dataB1 = {'T1': np.random.randint(20, 40, time_steps),
              'T2': np.random.randint(20, 50, time_steps),
              'Sin': np.sin(np.linspace(-3*np.pi, 3*np.pi, time_steps))*0.5,
              'Category Label 1': pd.Categorical(['B']*time_steps),
              'Category Label 2': pd.Categorical(['First']*time_steps),
              'Category Label 3': pd.Categorical(['10-20']*time_steps),
              }
    dataB2 = {'T1': np.random.randint(30, 50, time_steps),
              'T2': np.random.randint(30, 60, time_steps),
              'Category Label 1': pd.Categorical(['B']*time_steps),
              'Category Label 2': pd.Categorical(['Third']*time_steps),
              'Category Label 3': pd.Categorical(['20-30']*time_steps),
              }
    dataC1 = {'T1': np.random.randint(40, 60, time_steps),
              'T2': np.random.randint(40, 70, time_steps),
              'Cos': np.cos(np.linspace(-2*np.pi, 1*np.pi, time_steps))*0.66,
              'Sin': np.sin(np.linspace(-3*np.pi, 3*np.pi, time_steps))*0.75,
              'Category Label 1': pd.Categorical(['C']*time_steps),
              'Category Label 2': pd.Categorical(['Second']*time_steps),
              'Category Label 3': pd.Categorical(['20-30']*time_steps),
              }
    dataC2 = {'T1': np.random.randint(50, 70, time_steps),
              'T2': np.random.randint(50, 80, time_steps),
              'Cos': np.cos(np.linspace(-3*np.pi, 3*np.pi, time_steps))*0.33,
              'Category Label 1': pd.Categorical(['C']*time_steps),
              'Category Label 2': pd.Categorical(['Third']*time_steps),
              'Category Label 3': pd.Categorical(['20-30']*time_steps),
              }

    df = pd.concat([
        pd.DataFrame(data=dataA1, index=new_index),
        pd.DataFrame(data=dataA2, index=new_index),
        pd.DataFrame(data=dataB1, index=new_index),
        pd.DataFrame(data=dataB2, index=new_index),
        pd.DataFrame(data=dataC1, index=new_index),
        pd.DataFrame(data=dataC2, index=new_index),
        ])

    df.index.name = 'Time'
    df.reset_index(level=0, inplace=True)  # Make the index a regular column

    # With categories applied, e.g. sorting them could be enabled
    # Requires Pandas >= 0.21
#    c1 = CategoricalDtype(['A', 'B', 'C'], ordered=True)
#    c2 = CategoricalDtype(['First', 'Second', 'Third'], ordered=True)
#    df['Category Label 1'] = df['Category Label 1'].astype(c1)
#    df['Category Label 2'] = df['Category Label 2'].astype(c2)
#    print(df.sort_values(by=['Category Label 2', 'Category Label 1']))

#    df.to_excel('excel_text.xlsx')  # Save this as an Excel file if you want
#    print(df)  # Show the final DataFrame in the terminal window

    return df


def new_upload_button(save_path, callback, DatEx, label="Upload file"):
    '''
    A button widget that implements a special javascript callback function.
    This callback function makes the browser open a file dialog and allows
    the user to select a file, which is then uploaded to the folder 'save_path'
    on the server.

    Args:
        save_path (str) : Destination for the uploaded file.

        callback (func) : External function that is called on button press.

        label (str) : Button label text (optional).

    Return:
        button (widget) : Bokeh widget object, to be placed in a layout.

    '''
    def file_callback(attr, old, new):
        raw_contents = source.data['contents'][0]
        file_name = source.data['name'][0]
        # remove the prefix that JS adds
        prefix, b64_contents = raw_contents.split(",", 1)
        file_contents = base64.b64decode(b64_contents)
        file_path = os.path.join(save_path, file_name)

        with open(file_path, "wb") as f:
            f.write(file_contents)
        logging.debug("New file uploaded: " + file_path)
        callback(file_path, DatEx)

    # This 'source' will be filled by the JavaScript callback
    source = ColumnDataSource({'contents': [], 'name': []})
    source.on_change('data', file_callback)

    # Create the button widget
    button = Button(label=label, button_type="success")

    # The JavaScript magic is in this file
    code_path = os.path.join(os.path.dirname(__file__), 'models', 'upload.js')
    with open(code_path, 'r') as f:
        code_upload = f.read()

    # Connect the JavaScript code with the widget
    button.callback = CustomJS(args=dict(source=source), code=code_upload)

    return button


def create_heatmap(corr_matrix):
    '''
    Create and return a heatmap plot for a given correlation matrix.

    Args:
        corr_matrx (DataFrame) : A Pandas DataFrame produced by df.corr()

    Returns:
        p (figure) : A Bokeh figure containing the rectangle plot
    '''

    # Reformat the correlation matrix with melt() and use it as a Bokeh source
    corr_matrix.index.names = ['x']
    corr_df = corr_matrix.reset_index().melt(id_vars='x', var_name='y',
                                             value_name='value')
    source = ColumnDataSource(corr_df)

    # Construct a colourmap by combining two existing palettes
    colours = palettes.viridis(20)[9:19]
    colours2 = palettes.plasma(20)[9:19]
    colours.extend(reversed(colours2))
    mapper = LinearColorMapper(palette=colours, low=-1, high=1)

    # Create the figure
    p = figure(title='Correlation coefficient matrix',
               x_range=list(corr_matrix.columns),
               y_range=list(reversed(corr_matrix.columns)),
               x_axis_location='above', plot_width=800, plot_height=800,
               tools='save, pan, box_zoom, reset, wheel_zoom',
               toolbar_location='below', logo=None)

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    # Add the rectangle glyphs
    p.rect(x="x", y="y", width=1, height=1,
           source=source,
           fill_color={'field': 'value', 'transform': mapper},
           line_color=None)

    # Add the colour bar
    colour_bar = ColorBar(color_mapper=mapper,
                          major_label_text_font_size="7pt",
                          ticker=BasicTicker(desired_num_ticks=len(colours)),
                          # formatter=PrintfTickFormatter(format="%d%%"),
                          label_standoff=6, border_line_color=None,
                          location=(0, 0))
    p.add_layout(colour_bar, 'right')

    # Add the hover tool
    hover = HoverTool(point_policy='snap_to_data',
                      tooltips=[('x', '@x'),
                                ('y', '@y'),
                                ('corr', '@value')]
                      )
    p.add_tools(hover)

    return p
