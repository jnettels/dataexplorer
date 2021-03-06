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

Additional helper functions for the script dataexplorer.py
"""

import os
import logging
import base64
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, CustomJS, HoverTool
from bokeh.models import LinearColorMapper, BasicTicker, ColorBar
from bokeh.models.widgets import Button, TextInput
from bokeh.io import curdoc
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
              'Classification 1': pd.Categorical(['Class A']*time_steps),
              'Classification 2': pd.Categorical(['Class First']*time_steps),
              'Classification 3': pd.Categorical(['Class 10-20']*time_steps),
              }
    dataA2 = {'T1': np.random.randint(10, 30, time_steps),
              'T2': np.random.randint(10, 40, time_steps),
              'Classification 1': pd.Categorical(['Class A']*time_steps),
              'Classification 2': pd.Categorical(['Class Second']*time_steps),
              'Classification 3': pd.Categorical(['Class 10-20']*time_steps),
              }
    dataB1 = {'T1': np.random.randint(20, 40, time_steps),
              'T2': np.random.randint(20, 50, time_steps),
              'Sin': np.sin(np.linspace(-3*np.pi, 3*np.pi, time_steps))*0.5,
              'Classification 1': pd.Categorical(['Class B']*time_steps),
              'Classification 2': pd.Categorical(['Class First']*time_steps),
              'Classification 3': pd.Categorical(['Class 10-20']*time_steps),
              }
    dataB2 = {'T1': np.random.randint(30, 50, time_steps),
              'T2': np.random.randint(30, 60, time_steps),
              'Classification 1': pd.Categorical(['Class B']*time_steps),
              'Classification 2': pd.Categorical(['Class Third']*time_steps),
              'Classification 3': pd.Categorical(['Class 20-30']*time_steps),
              }
    dataC1 = {'T1': np.random.randint(40, 60, time_steps),
              'T2': np.random.randint(40, 70, time_steps),
              'Cos': np.cos(np.linspace(-2*np.pi, 1*np.pi, time_steps))*0.66,
              'Sin': np.sin(np.linspace(-3*np.pi, 3*np.pi, time_steps))*0.75,
              'Classification 1': pd.Categorical(['Class C']*time_steps),
              'Classification 2': pd.Categorical(['Class Second']*time_steps),
              'Classification 3': pd.Categorical(['Class 20-30']*time_steps),
              }
    dataC2 = {'T1': np.random.randint(50, 70, time_steps),
              'T2': np.random.randint(50, 80, time_steps),
              'Cos': np.cos(np.linspace(-3*np.pi, 3*np.pi, time_steps))*0.33,
              'Classification 1': pd.Categorical(['Class C']*time_steps),
              'Classification 2': pd.Categorical(['Class Third']*time_steps),
              'Classification 3': pd.Categorical(['Class 20-30']*time_steps),
              }

    df = pd.concat([
        pd.DataFrame(data=dataA1, index=new_index),
        pd.DataFrame(data=dataA2, index=new_index),
        pd.DataFrame(data=dataB1, index=new_index),
        pd.DataFrame(data=dataB2, index=new_index),
        pd.DataFrame(data=dataC1, index=new_index),
        pd.DataFrame(data=dataC2, index=new_index),
        ], sort=True)

    df.index.name = 'Time'
    df.reset_index(level=0, inplace=True)  # Make the index a regular column

    # With categories applied, e.g. sorting them could be enabled
    # Requires Pandas >= 0.21
#    c1 = CategoricalDtype(['Class A', 'Class B', 'Class C'], ordered=True)
#    c2 = CategoricalDtype(['Class First', 'Class Second', 'Class Third'],
#                          ordered=True)
#    df['Classification 1'] = df['Classification 1'].astype(c1)
#    df['Classification 2'] = df['Classification 2'].astype(c2)
#    print(df.sort_values(by=['Classification 2', 'Classification 1']))

#    df.to_excel('excel_text.xlsx')  # Save this as an Excel file if you want
#    print(df)  # Show the final DataFrame in the terminal window

    return df


def new_upload_button(save_path, callback, DatEx, label="Upload file"):
    '''Return a upload button widget that implements a special JavaScript
    callback function.
    This callback function makes the browser open a file dialog and allows
    the user to select a file, which is then uploaded to the folder 'save_path'
    on the server.

    Args:
        save_path (str) : Destination for the uploaded file.

        callback (func) : External function that is called on button press.

        DatEx (Dataexplorer): The object containing all the session information

        label (str, optional) : Button label text.

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

        try:
            with open(file_path, "wb") as f:
                f.write(file_contents)
        except Exception as ex:
            logging.exception(ex)
        else:
            logging.debug("New file uploaded: " + file_path)
        callback(file_path, DatEx)

    # This 'source' will be filled by the JavaScript callback
    source = ColumnDataSource({'contents': [0], 'name': [0],
                               'server_mode': [DatEx.server_mode]})
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


def new_download_button(DatEx, label='Download current data selection'):
    '''Return a download button widget that implements a special JavaScript
    callback function.
    This callback function makes the browser open a file dialog and allows
    the user to download the currently selected data as a .csv file.

    TODO (Maybe) Remove separator from column names

    Args:
        DatEx (Dataexplorer): The object containing all the session information

        label (str, optional) : Button label text.

    Return:
        button (widget) : Bokeh widget object, to be placed in a layout.

    '''
    sel = DatEx.source.selected
    if sel is None or sel.indices == []:
        df_sel = DatEx.df
    else:
        df_sel = DatEx.df.iloc[sel.indices]

    filename = 'DataExplorer_Download.csv'
    filetext = df_sel.to_csv(sep=';',
                             columns=DatEx.get_columns_sorted(),
                             index=False,
                             )
    filetype = 'text/csv;charset=utf-8;'

    source = ColumnDataSource({'filename': [filename],
                               'filetext': [filetext],
                               'filetype': [filetype],
                               })

    # Create the button with the connected JavaScript code callback
    button = Button(label=label, button_type="success", name='download_button')
    # The JavaScript magic is in this file
    f_path = os.path.join(os.path.dirname(__file__), 'models', 'download.js')
    with open(f_path, 'r') as f:
        code_download = f.read()
    button.callback = CustomJS(args=dict(source=source), code=code_download)

    return button


def show_confirm_dialog(message):
    '''Function for testing a confirmation dialog. The Python code does not
    wait for the answer. Instead the source_callback is called when the choice
    is made.
    '''
    def source_callback(attr, old, new):
        '''Something meaningful needs to be put in here. Made call another
        callback, as in new_download_button.
        '''
        choice = source.data['choice'][0]
        curdoc().remove_root(ti_code)
        print(choice)

    source = ColumnDataSource({'message': [message], 'choice': [None]})
    source.on_change('data', source_callback)

    # The JavaScript magic is in this file
    f_path = os.path.join(os.path.dirname(__file__), 'models', 'confirm.js')
    with open(f_path, 'r') as f:
        code_confirm = f.read()
    ti_code = TextInput(value='')
    ti_code.js_on_change('value', CustomJS(args=dict(source=source),
                                           code=code_confirm))
    curdoc().add_root(ti_code)
    ti_code.value = ' '  # Trigger the JavaScript code

    return


def enable_responsiveness(DatEx, tabs):
    '''Call this function once to enable resizing of some of the UI elements.
    This does not make the Bokeh app truly responsive, but at least makes
    the grid of figures and the DataTable adapt to changes of the browser
    window size, whenever the user switches between the tabs.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

        tabs (Bokeh widget): A Bokeh 'Tabs' widget to attach the callback to

    Returns:
        None
    '''
    def source_callback(attr, old, new):
        '''Callback of the source object. Updates the properties of DatEx with
        the new browser sizes. It will only be called if the sizes change.
        '''
        DatEx.window_height = source.data['window_height'][0]
        DatEx.window_width = source.data['window_width'][0]
        # Resize the DataTable
        DatEx.data_table.width = DatEx.window_width-50
        DatEx.data_table.height = DatEx.window_height-100
        # Mark the grid of figures for an update (which will use the new size)
        DatEx.grid_needs_update = True
        # Adapt scrollable columns on settings page
        DatEx.cg_vals_col.height = max(DatEx.window_height-410, 415)
        DatEx.cg_classifs_col.height = DatEx.cg_vals_col.height
        # Resize the correlation matrix
        matrix_size = min(DatEx.window_height-100, DatEx.window_width-20)
        DatEx.corr_matrix_heatmap.width = matrix_size
        DatEx.corr_matrix_heatmap.height = matrix_size
        return

    # Create source object that will react to a change of its data
    source = ColumnDataSource({'window_height': [DatEx.window_height],
                               'window_width': [DatEx.window_width]})
    source.on_change('data', source_callback)

    # Define JavaScript code and attach it to the widget
    js_code = '''source.data = {'window_height': [window.innerHeight],
                                'window_width': [window.innerWidth]};
                 source.change.emit()'''
    tabs.js_on_change('active', CustomJS(args=dict(source=source),
                                         code=js_code))
    return


def create_heatmap(corr_matrix, DatEx):
    '''Create and return a heatmap plot for a given correlation matrix.

    Args:
        corr_matrx (DataFrame) : A Pandas DataFrame produced by df.corr()

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        p (figure) : A Bokeh figure containing the rectangle plot
    '''

    # Reformat the correlation matrix with melt() and use it as a Bokeh source
    corr_matrix.index.names = ['x']
    corr_df = corr_matrix.reset_index().melt(id_vars='x', var_name='y',
                                             value_name='value')
    DatEx.hm_source = ColumnDataSource(corr_df)

    # Construct a colourmap by combining two existing palettes
    colours = palettes.viridis(20)[9:19]
    colours2 = palettes.plasma(20)[9:19]
    colours.extend(reversed(colours2))
    mapper = LinearColorMapper(palette=colours, low=-1, high=1)
    matrix_size = min(DatEx.window_height-100, DatEx.window_width-20)

    # Create the figure
    p = figure(  # title='Correlation coefficient matrix',
               x_range=list(corr_matrix.columns),
               y_range=list(reversed(corr_matrix.columns)),
               x_axis_location='above', plot_width=matrix_size,
               plot_height=matrix_size,
               tools='save, pan, wheel_zoom, box_zoom, reset, tap',
               active_drag=None,
               toolbar_location='below',
               name='corr_matrix_heatmap')

    p.toolbar.logo = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "7pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    # Add the rectangle glyphs
    p.rect(x="x", y="y", width=1, height=1,
           source=DatEx.hm_source,
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

    # Hack: Force the figure to redraw when its height changes
    js_code = '''var event = document.createEvent('HTMLEvents');
                 event.initEvent('resize');
                 window.dispatchEvent(event);'''
    p.js_on_change('height', CustomJS(code=js_code))

    return p


def read_filetypes(filepath):
    '''Read any file type with stored data and return the Pandas DataFrame.
    Wrapper around Pandas' read_excel() and the local read_csv_formats().

    For Excel files, missing values (empty cells) are allowed and normally
    filled with ``NaN``.
    However: If all but the first cells of a row are empty, we try to create
    a multiindex from the headers in the rows above that one.
    This should match the structure created when saving a multiindex DataFrame
    with ``pd.DataFrame.to_excel()``.

    ================  ======  ======  ======  ======
    Classification 1  A               B
    Classification 2  First   Second  First   Second
    Time
    00:00             0       100     5       110
    00:15             1       90      4       120
    00:30             2       80      5       115
    00:45             1       70      7       110
    ================  ======  ======  ======  ======
    '''
    filetype = os.path.splitext(os.path.basename(filepath))[1]
    if filetype in ['.xlsx', '.xls']:
        # Excel can be read automatically with Pandas
        df_new = pd.read_excel(filepath)

        # Find all rows with missing values
        rows_with_NaN = df_new.isnull().any(axis=1)
        # Try to find the headers of a multiindex
        header_rows = []  # Default: No multiindex
        for i in df_new[rows_with_NaN].index:
            row = df_new.loc[i, :]
            # The last row of the multindex is defined as one where all cells
            # (except the first one) contain NaN.
            if row[1:].isnull().all():
                header_rows = list(range(0, i + 1))

        # If we assume there is a multiindex, try again to read the Excel file
        # Then we flatten that index by stacking and reordering.
        if len(header_rows) != 0:
            df_new = pd.read_excel(filepath, header=[header_rows], index_col=0)
            df_new = df_new.stack(level=header_rows[:-1])
            df_new = df_new.reorder_levels(header_rows[1:]+[header_rows[0]])
            df_new = df_new.sort_index()
            df_new.reset_index(inplace=True)

    elif filetype in ['.csv']:
        # csv files can have different formats
        df_new = read_csv_formats(filepath)  # My own wrapper around Pandas
    elif filetype in ['.dat', '.txt', '.out']:
        logging.warning('Unsupported file extension: ' + filetype +
                        '. Trying to read it like a csv file.')
        df_new = read_csv_formats(filepath)
    else:
        raise NotImplementedError('Unsupported file extension: '+filetype)

    return df_new


def read_csv_formats(filepath):
    ''' Read a csv file and return a Pandas DataFrame. This is a wrapper
    around Pandas' read_csv() where the format of the csv file is determined
    first. The possible formats are:
        - 'synavision CSV format 2.0 DE': As exported by the Digital Testbench
        - Standard: Here we try to guess the format rules. Files that follow
            the Excel rules should work.
    Args:
        filepath (string) : The path to the file

    Returns:
        df (DataFrame) : A Pandas Dataframe

    '''
    with open(filepath, 'r') as f:
        first_line = f.readline()  # We need the first line to get the format

    if 'synavision CSV format 2.0 DE' in first_line:
        # The Digital Testbench produces a special header that we deal with
        df = pd.read_csv(filepath,
                         sep=';', engine='python',
                         decimal=',', thousands='.',
                         header=[1, 6],  # Row 1: Names; Row 6: Units
                         parse_dates=[0],  # Parse first column as date
                         infer_datetime_format=True,
                         dayfirst=True,  # DD.MM.YYYY
                         )

        # We created a multi-index that has the units on its second level
        units = df.columns.get_level_values(1).values
        for i, unit in enumerate(units):
            if 'Unnamed' in unit or 'unit' in unit:
                units[i] = '-'  # Replace empty units with '-'
        # Drop the level with the units
        df.columns = df.columns.droplevel(1)

        # Create a dict with the old_index and new_index, where column names
        # and units are combined with the pattern: 'Name [Unit]'
        old_index = df.columns.values
        new_index = [x+' ['+y+']' for x, y in zip(old_index, units)]
        cols_dict = dict(zip(old_index, new_index))

        # Rename the columns with the dict, then rename the time column again
        df.rename(columns=cols_dict, inplace=True)
        df.rename(columns={'id [-]': 'Time'}, inplace=True)

    else:
        # Standard format: Here we guess everything. May or may not work
        df = pd.read_csv(filepath,
                         sep=None, engine='python',  # Guess separator
                         parse_dates=[0],  # Try to parse first column as date
                         infer_datetime_format=True)

    return df
