# -*- coding: utf-8 -*-
'''
@version: 1.0.0

@author: Joris Nettelstroth

The DataExplorer should help you explore correlations within your data. It
features a user interface that shows scatter plots of all the variables in
your data. Classes found in your data can be used to filter the views.

You can start this program with the "start_bokeh.cmd" batch file. This starts
a Bokeh server that can be accessed from any computer in the network.

Alternatively, you can start it with your own command prompt in Windows:
    - Go to Start and type "cmd"
    - Select "Eingabeaufforderung"
    - Change to the directory containing the folder "dataexplorer" (the folder
      including the files "main.py" and "dataexplorer.py") with the command:
      "cd path/to/folder"
    - Start a Bokeh server running this application by typing:
      "bokeh serve dataexplorer --show"
    - Your webbrowser should open and display the interface of the program
    - Under settings, hit the button to load your own Excel file

The file excel_example.xlsx shows the required input format and gives hints
about the usage.

If you do not yet have Python and Bokeh installed, the easiest way to do that
is by downloading and installing "Anaconda" from here:
https://www.anaconda.com/download/
It's a package manager that distributes Python with data science packages.

During installation, please allow to add variables to $PATH (or do that
manually afterwards.) This allows Bokeh to be started from everywhere, which
is required for the batch file to work.

TODO:
    - Transfer session settings via DatEx.__dict__
    - Include download button for correlation matrix
    - Include download for plots

Known issues:
    - Plots fail when Time column includes 'NaT', so those columns are removed
    - By default, there is a limit of about 8 MB of data upload to the
      browser. This is controlled by the Tornado server, with the parameter
      'websocket_max_message_size'.
      In Anaconda3/Lib/site-packages/bokeh/server/tornado.py go to line 221:

      .. code:: python

          super(BokehTornado, self).__init__(all_patterns)

      For 100 MB upload limit, replace it with:

      .. code:: python

          super(BokehTornado, self).__init__(all_patterns,
                websocket_max_message_size=100*1024*1024)

      Also see: https://github.com/bokeh/bokeh/issues/7374

'''

import pandas as pd
import numpy as np
import itertools
import os
import re
import logging
import bokeh
import yaml  # Read YAML configuration files
import unicodedata
from bokeh.layouts import widgetbox, gridplot, layout
from bokeh.layouts import row
from bokeh.layouts import column
from bokeh.models.widgets import CheckboxButtonGroup, Select, CheckboxGroup
from bokeh.models.widgets import Div, DataTable, TableColumn, DateFormatter
from bokeh.models.widgets import Panel, Tabs, TextInput, Slider, Toggle
from bokeh.models.widgets import RadioGroup, Button
from bokeh.models import ColumnDataSource  # , CategoricalColorMapper
from bokeh.models import CustomJS, HoverTool, Span, Selection
# from bokeh.models import CDSView, BooleanFilter, GroupFilter
from bokeh.plotting import figure
from bokeh import palettes
from bokeh.io import curdoc
from functools import partial
from pandas.api.types import is_categorical_dtype, CategoricalDtype
from bokeh.io import export_png, export_svgs
from distutils.version import LooseVersion

import holoviews as hv
import holoviews.operation.datashader as hd
import datashader as ds
import datashader.transfer_functions as tf

# My own library of functions from the file helpers.py
from helpers import (new_upload_button, create_test_data, create_heatmap,
                     read_filetypes, new_download_button,
                     enable_responsiveness)

# Global Pandas option for displaying terminal output
# pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 0)  # Fit number of columns to terminal

# Check some version requirements
pd_v_required = '0.23.0'
if LooseVersion(pd.__version__) < LooseVersion(pd_v_required):
    logging.critical('Warning: Pandas version '+pd_v_required+' is required.' +
                     ' Your version is '+pd.__version__)
bk_v_required = '1.0.1'
if LooseVersion(bokeh.__version__) < LooseVersion(bk_v_required):
    logging.critical('Warning: Bokeh version '+bk_v_required+' is required.' +
                     ' Your version is '+bokeh.__version__)


class Dataexplorer(object):
    '''The Dataexplorer class is the central element of the structure of this
    application. By creating a Dataexplorer object, all tasks necessary to
    produce the Bokeh user interface are performed.
    Furthermore, the Dataexplorer object stores all the settings and properties
    of the current session of the application.
    Most functions in this Python script only take the Dataexlorer object
    as an input parameter and can manipulate its properties directly.
    When new data is loaded, a new Dataexplorer object must be created to
    recreate the user interface. Certain settings can be handed over to the
    new object to make them survive the switch of sessions.
    '''

    def __init__(self, df, data_name, server_mode, combinator=0, vals_max=6,
                 window_height=974, window_width=1920,
                 output_backend='webgl', render_mode='Bokeh'):
        '''Return a Dataexplorer object, the object containing all the session
        information. Initialize all object properties.
        Perform all the tasks necessary to create the Data Explorer user
        interface by calling the required functions.

        Args:
            df (Pandas DataFrame): The input data we want to explore.

            data_name (str): The filename of the current data set.

            server_mode (bool): Is server_mode enabled or disabled?

            combinator (int, optional): A identifier for combinatoric generator

            vals_max (int, optional): A threshold for number of value columns

            window_height (int, optional): Browser window height

            window_width (int, optional): Browser window width

            output_backend (str, optinal): Bokeh's rendering backend
            (``"canvas``", ``"webgl"`` or ``"svg"``)

        Returns:
            None
        '''
        # Settings that are exposed in the config file:
        self.vals_max = vals_max  # Threshold for number of value columns
        self.server_mode = server_mode
        self.render_mode = render_mode
        self.combinator = combinator  # Identifier for combinatoric generator
        self.circle_size = 5
        self.p_h = 250  # global setting for plot_height
        self.p_w = 250  # global setting for plot_width
        self.load_mode_append = 0  # 0 equals False equals replace
        self.window_height = window_height  # Pixels of browser window
        self.window_width = window_width  # Pixels of browser window
        self.output_backend = output_backend
        self.palette = get_palette_default()  # List of colors
        self.palette_large_name = False  # Large backup palette, e.g. 'plasma'
        self.colourmap_user = dict()
        self.export_corr_matrix = False
        self.HoverTool_enable = True

        if self.server_mode is False:
            perform_config(self)  # Save or load the config file

        # More elements of 'self' that are not exposed to the config file:
        self.df = df
        self.data_name = data_name
        self.selected_figs = []
        self.hm_sel_ids = []
        self.grid_needs_update = False
        self.table_needs_update = False
        self.corr_matrix_needs_update = False
        self.classifs_need_update = False

        # Set classifications, their classes and value column names
        try:
            analyse_dataframe(self)
        except Exception as ex:  # Is thrown if the df has an incorrect format
            show_info(str(ex))
            return  # Skip the rest in case of an exception

        # Use Bokeh to plot the data in an interactive way
        create_plots(self)

        # Prepare the filters used to explore the data
        prepare_filter(self)

        # Create and get a list of the widgets for tab 'Scatters'
        create_widgets_1(self)

        # Create and get a list of the widgets for tab 'Settings'
        create_widgets_2(self)

        # Create and get the DataTable for tab 'DataTable'
        create_data_table(self)

        # Create a correlation coefficient matrix plot for tab 'Correlation'
        create_corr_matrix_heatmap(self)

        # Create a Bokeh 'layout' from all the tabs
        create_layout(self)

        # Loading the file enabled the loading animation. Now we disable it:
        toggle_loading_mouse(False)

    def get_columns_sorted(self):
        '''Return a sorted list of the active column names in the order:

            - Time column (if present)
            - Classification columns
            - Value columns

        Args:
            self (Dataexplorer): The object containing all the information

        Returns:
            columns (list): Sorted list of column names
        '''
        # If a time column exists, it should be the first column in DataTable
        if self.col_time is not None and self.col_time in self.vals_active:
            vals_list = self.vals_active.copy()
            vals_list.remove(self.col_time)
            columns = [self.col_time] + self.classifs_active + vals_list
        else:
            columns = self.classifs_active + self.vals_active

        return columns


def perform_config(self):
    '''All configuration settings can be accessed by the user if they are
    running a local server (``server_mode == False``) with the help of a YAML
    config file. If no config file exists, it is created with the current
    settings. If it exists, the settings are loaded.

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    config_file = os.path.join(os.path.dirname(__file__),
                               'templates/config.yaml')

    config = self.__dict__
    config['0 Info'] = \
        ['This is a YAML configuration file for the DataExplorer',
         'You can use "#" to comment out lines in this file',
         'To restore the original config, just delete this file and ' +
         'restart the DataExplorer',
         'The colours of the palette are defined as hex values. For help see',
         'https://www.w3schools.com/colors/colors_picker.asp',
         'With "colourmap_user" you can define individual colours for ' +
         'specific class names, with priority over the palette']

    if not os.path.exists(config_file):
        yaml.dump(config, open(config_file, 'w'), default_flow_style=False)
    else:
        try:
            config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

            for key, value in config.items():
                self.__dict__[key] = value
        except Exception as ex:
            logging.error(str(ex))


def analyse_dataframe(self):
    '''Analyse a given DataFrame to separate the columns in classes and
    values. "Object" columns become class columns, their column names are saved
    as classifications.

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''

    df = self.df

    columns = df.columns.values.tolist()
    classifs = []  # List of the classifications (columns that contain classes)
    vals = []  # List of the column names that contain values
    self.col_time = None
    for column_ in columns:
        # The column contains classes or values
        if df[column_].dtype == object or is_categorical_dtype(df[column_]):
            classifs.append(column_)  # Classification found
            df[column_] = list(map(str, df[column_]))  # Convert to strings
        elif '!' in column_:
            # column_new = column_.replace('!', '')
            # df.rename(columns={column_: column_new}, inplace=True)
            # column_ = column_new
            df[column_] = list(map(str, df[column_]))
            classifs.append(column_)  # Classification found
        else:
            if df[column_].dtype == 'datetime64[ns]':
                if pd.NaT in df[column_].tolist():
                    # 'Not a Time' in a time column makes the plots not render
                    # properly, so we need to sort those columns out
                    show_info('Warning: Column "'+column_+'" removed from ' +
                              'data, because of missing entries.')
                    df.drop(columns=[column_], inplace=True)
                else:
                    self.col_time = column_  # Save column as time column
                    vals.append(column_)  # Value column found
            else:
                vals.append(column_)  # Value column found

    if classifs == []:  # No classification were found
        # This is not an ideal use case, but still possible
        show_info('Info: No classification columns found in the file' +
                  ', using the filename as a class instead! Please ' +
                  'refer to example Excel file for instructions.')
        df['File'] = [self.data_name]*len(df)  # Fall back to a default
        classifs = ['File']
    if vals == []:  # No value column were found
        # This cannot be accepted
        raise LookupError('Error: No value columns found in the file! Please' +
                          ' refer to example Excel file for instructions.')

    classes_dict = dict()  # A dict with the classes of each classification
    for classif in classifs:
        try:  # Try to sort the classes
            classes = list(set(df[classif]))  # Get classes of classification
            try:  # First try sorting like a number
                classes.sort(key=float)
            except Exception:  # Then try default sorting
                logging.debug("Debug: Float sorting failed")
                classes.sort()
            classes_dict[classif] = classes  # Store sorted classes
            # Make ordered 'categorical' by using the sorted classes
            cat_Dtype = CategoricalDtype(categories=classes, ordered=True)
            df[classif] = df[classif].astype(cat_Dtype)
        except Exception as ex:  # Map to strings before sorting
            # TODO: Check if this is still required
            logging.error("Debug: Regular sorting failed")
            df[classif] = list(map(str, df[classif]))
            classes = list(set(df[classif]))  # Get classes of classification
            classes_dict[classif] = sorted(classes)  # Sort and store them
            pass
#        print(classif, classes_dict[classif])

    if len(vals) > self.vals_max:
        vals_active = vals[:self.vals_max]  # Use a shortened list of vals
    else:
        vals_active = vals

    if self.combinator == 4 and self.col_time is None:
        # If combinator 4 'Time series only' was selected in the config, but
        # the current data has no time column, we must reset the combinator
        self.combinator = 0

    self.vals = vals
    self.classifs = classifs
    self.classes_dict = classes_dict
    self.vals_active = vals_active  # active value columns
    self.classifs_active = classifs  # active classification columns

    # The first classification is the default colour classification
    self.colour_classif = classifs[0]  # Name of current colour classification
    # Create (or update) 'Legend' and 'Colours' columns and sort data
    update_colours(self)
    # Create the Bokeh ColumnDataSource object from Pandas DataFrame
    self.source = ColumnDataSource(data=df)
    # Initialize empty selection
    self.source.selected = Selection(indices=[])

    return


def create_plots(self):
    '''Use Bokeh to plot the data in an interactive way. The Bokeh settings
    are defined in this function, as well as the combinatoric generator which
    generates the combinations of all the 'vals'. For each combination, a
    figure is created and all of those figures are combined into one grid.

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        grid (Gridplot) : Grid containing all created figures

    '''

#    view = CDSView(source=source, filters=[])  # Create an empty view object

    plot_set = {'tools': ['pan', 'wheel_zoom', 'box_zoom', 'reset',
                          'lasso_select', 'box_select', 'save', 'help'],
                # 'active_scroll': 'wheel_zoom',
#                'active_inspect': None,
                'plot_height': self.p_h, 'plot_width': self.p_w,
                'output_backend': self.output_backend,
                }
    glyph_set = {'color': 'Colours', 'hover_color': 'Colours',
                 'fill_alpha': 0.2, 'hover_alpha': 1,
                 'size': self.circle_size,
                 # Set the properties of glyphs that are not selected
                 'nonselection_fill_color': 'grey',
                 'nonselection_fill_alpha': 0.1,
                 'nonselection_line_color': None,
                 }
    span_set = {'location': 0, 'line_color': 'grey',
                'line_dash': 'dashed', 'line_width': 1}

    self.selected_figs = get_combinations(self)

    self.fig_list = []  # List with the complete figures
    self.glyph_list = []  # List of GlyphRenderers
    self.span_list = []  # List of spans (coordinate center lines)
    self.cvs_list = []  # List of DataShadar canvases

    for x_val, y_val in self.selected_figs:
        # Is x_axis or y_axis of data type 'datetime'?
        x_time = (self.df[x_val].dtype == 'datetime64[ns]')
        y_time = (self.df[y_val].dtype == 'datetime64[ns]')

        # Prepare the HoverTool options:
        formatters_dict = {}
        tips_list = [[classif, '@{'+classif+'}'] for classif in self.classifs]

        # DateTime columns require some very special treatment
        strftime = '%y-%m-%d %H:%M:%S'  # date and time format
        if x_time and y_time:
            p = figure(x_axis_type='datetime', y_axis_type='datetime',
                       **plot_set)
            formatters_dict[x_val] = 'datetime'
            formatters_dict[y_val] = 'datetime'
            tips_list.append([x_val, '@{'+x_val+'}{'+strftime+'}'])
            tips_list.append([y_val, '@{'+y_val+'}{'+strftime+'}'])
        elif x_time:
            p = figure(x_axis_type='datetime', **plot_set)
            formatters_dict[x_val] = 'datetime'
            tips_list.append([x_val, '@{'+x_val+'}{'+strftime+'}'])
            tips_list.append([y_val, '@{'+y_val+'}'])
        elif y_time:
            p = figure(y_axis_type='datetime', **plot_set)
            formatters_dict[y_val] = 'datetime'
            tips_list.append([x_val, '@{'+x_val+'}'])
            tips_list.append([y_val, '@{'+y_val+'}{'+strftime+'}'])
        else:
            p = figure(**plot_set)
            tips_list.append([x_val, '@{'+x_val+'}'])
            tips_list.append([y_val, '@{'+y_val+'}'])

        # Set the text labels of the axis
        p.xaxis.axis_label = x_val
        p.yaxis.axis_label = y_val
        p.yaxis.major_label_orientation = "vertical"
        p.name = x_val + '; ' + y_val  # give the plot a name

        if self.render_mode == 'Bokeh':
            # Create the actual circle GlyphRenderer
            cr = p.circle(x=x_val, y=y_val, source=self.source, **glyph_set)
            self.glyph_list.append(cr)

            # Create HoverTool:
            hover = HoverTool(point_policy='follow_mouse',  # 'snap_to_data',
                              tooltips=tips_list,
                              renderers=[cr],  # Uses 'hover_*' options
                              formatters=formatters_dict)
            if self.HoverTool_enable:
                p.add_tools(hover)  # Not enabling hover boosts performance

        if self.render_mode == 'HoloViews':
            # HoloViews
#            hv.extension('bokeh')
#            points = hv.Scatter(self.df, kdims=[x_val, y_val])
            points = hv.Points(self.df, kdims=[x_val, y_val])

#            dataset = hv.Dataset(self.df)
#            points = dataset.to(hv.Points, [x_val, y_val],
#                                groupby='Legend').overlay()
#            points.opts({'Points': plot_set})
#            points = hd.datashade(points)
            points = hd.datashade(points, aggregator=ds.count_cat('Legend'))

            # BUG: The return of dynspread cannot be retrieved with get_plot
#            points = hd.dynspread(points, threshold=0.50, how='over')

            renderer = hv.renderer('bokeh').instance(mode='server')
            hvplot = renderer.get_plot(points, curdoc())
#            hvplot = hv.plotting.bokeh.BokehRenderer.get_plot(points)
#            hvplot.set_param(tools=['lasso_select'])
            p = hvplot.state
#            p = points
#            cr = p.renderers[6]
#            print(cr)

        if self.render_mode == 'DataShader':
            df_copy = self.df.copy()

            # Prepare the axis ranges, converting DateTime to float
            ref = pd.datetime(1970, 1, 1, 00, 00, 00)
            if self.df[x_val].dtype == 'datetime64[ns]':
                df_copy[x_val] = (df_copy[x_val] - ref)/np.timedelta64(1, 'ms')
            if self.df[y_val].dtype == 'datetime64[ns]':
                df_copy[y_val] = (df_copy[y_val] - ref)/np.timedelta64(1, 'ms')

            x_range = [df_copy[x_val].min(), df_copy[x_val].max()]
            y_range = [df_copy[y_val].min(), df_copy[y_val].max()]

            # Contruct the DataShader image
            cvs = ds.Canvas(plot_width=self.p_w, plot_height=self.p_h,
                            x_range=x_range, y_range=y_range)
            self.cvs_list.append(cvs)

            try:
                agg = cvs.points(df_copy, x_val, y_val,
                                 agg=ds.count_cat('Legend'))
                img = tf.shade(agg, color_key=self.colormap, how='eq_hist',
                               min_alpha=165)
                img = tf.dynspread(img, threshold=0.99)
                p.image_rgba(image=[img.data], x=x_range[0], y=y_range[0],
                             dw=x_range[1]-x_range[0],
                             dh=y_range[1]-y_range[0])
                p.x_range.start = x_range[0]
                p.x_range.end = x_range[1]
                p.y_range.start = y_range[0]
                p.y_range.end = y_range[1]

            except ValueError and ZeroDivisionError as ex:
                logging.error(ex)
                p.image_rgba(image=[], x=x_range[0], y=y_range[0],
                             dw=x_range[1]-x_range[0],
                             dh=y_range[1]-y_range[0])
                p.x_range.start = None
                p.x_range.end = None
                p.y_range.start = None
                p.y_range.end = None
                pass
            except Exception as ex:
                logging.exception(ex)
                p.image_rgba(image=[], x=[0, 1], y=[0, 1], dw=1, dh=1)
                pass


        # Testing: Allow zooming in datashader
#        print(p.x_range.start)
#        p.on_change('x_range', datashader_callback)
#        print(p.__dict__)
#        p.x_range.on_change('start', partial(datashader_callback, p=p, DatEx=self, x_val=x_val, y_val=y_val))
#        p.x_range.on_change('end', partial(datashader_callback, p=p, DatEx=self, x_val=x_val, y_val=y_val))
#        p.y_range.on_change('start', partial(datashader_callback, p=p, DatEx=self, x_val=x_val, y_val=y_val))
#        p.y_range.on_change('end', partial(datashader_callback, p=p, DatEx=self, x_val=x_val, y_val=y_val))
#        print(p.x_range.__dict__)
#        p = update_datashader(self, p, x_val, y_val)
#        break

        # Add figure to list of all figures
        self.fig_list.append(p)

        # Add horizontal and vertical lines in the center coordinates
        span_h = Span(**span_set, dimension='height', level='underlay')
        span_w = Span(**span_set, dimension='width', level='underlay')
        self.span_list.append(span_h)
        self.span_list.append(span_w)
        p.add_layout(span_h)
        p.add_layout(span_w)

#        # DataShader
#        def image_callback(x_range, y_range, w, h):
#            cvs = ds.Canvas(plot_width=w, plot_height=h,
#                            x_range=x_range, y_range=y_range)
#            agg = cvs.points(self.df, x_val, y_val, ds.count_cat('Legend'))
#            img = tf.shade(agg)
#            return tf.dynspread(img, threshold=0.80)
#
#        ds.bokeh_ext.InteractiveImage(p, image_callback)

    '''
    The plots are completed, now we add two figures for the legends that go to
    the top and bottom of the page. For a nice look, we remove all parts of the
    figures but the legends themselves.
    '''
    max_width = self.window_width-75
    legend_top = figure(plot_height=50, plot_width=max_width)
    legend_bot = figure(plot_height=2*self.p_h, plot_width=2*self.p_w)

    for legend_x in [legend_top, legend_bot]:
        legend_x.circle(x=self.vals[0], y=self.vals[1], source=self.source,
                        **glyph_set, legend='Legend', visible=False)
        legend_x.toolbar_location = None
        legend_x.legend.location = 'top_left'
        legend_x.legend.margin = 0
        legend_x.axis.visible = False
        legend_x.grid.visible = False
        legend_x.outline_line_color = None

    legend_top.legend.orientation = 'horizontal'
    legend_top.name = 'legend_horizontal'
    legend_bot.name = 'legend_vertical'
#    self.fig_list.append(legend_bot)  # Bottom legend currently disabled

    # Get the number of grid columns from the rounded square root of number of
    # figures.
    if self.combinator == 2:
        n_grid_cols = int(np.sqrt(len(self.fig_list)))
    elif self.combinator == 1:
        n_grid_cols = int(np.floor(np.sqrt(len(self.fig_list))))
    else:
        n_grid_cols = int(round(np.sqrt(len(self.fig_list)))) + 1

    # Reduce the number of grid columns if the grid is too large
    while n_grid_cols*self.p_w + 20 > max_width:  # Includes scrollbar
        n_grid_cols -= 1
    n_grid_cols = max(n_grid_cols, 1)  # must not become zero

    # Create the final grid of figures
    grid = gridplot(self.fig_list, ncols=n_grid_cols, toolbar_location='left',
                    toolbar_options={'logo': None})

    '''Make the plots scrollable'''
    # The children of 'grid' are the ToolbarBox [0] and a column containing
    # all the rows of plots [1].
    # We assign this column the CSS class 'scrollable'. Together with the style
    # added to index.html, this allows the gridplot to become a scrollable box.
    grid.children[1].css_classes = ['scrollable']
    # We also need to fix the boundaries of this html DIV. The scrollbar
    # appears when the contents are too large (overflow occurs).
    grid.children[1].sizing_mode = 'fixed'
    grid.children[1].height = self.window_height-60
    grid.children[1].width = max_width

    self.grid = grid
    self.legend_top = legend_top
    self.legend_bot = legend_bot

    # Link the ranges of the axis with the same description
    link_axis_ranges(self)

    return self.grid


def link_axis_ranges(self):
    '''Link the figure ranges of the axis with the same description.

    Args:
        self (Dataexplorer) : The object containing all the session information

    Returns:
        None
    '''
    # Create and fill dicts with reference figures for each column
    ref_figs_x = dict()  # Reference figure to link x_axis to
    ref_figs_y = dict()  # Reference figure to link y_axis to
    for i, vals in enumerate(self.selected_figs):
        x_val, y_val = vals
        ref_figs_x[x_val] = i  # Store i as reference figure
        ref_figs_y[y_val] = i  # Store i as reference figure

    # Walk through the list of figures to link each column to their
    # reference figure
    for i, vals in enumerate(self.selected_figs):
        x_val, y_val = vals
        fig = self.fig_list[i]
        # Set x and y range of current figure
        fig.x_range = self.fig_list[ref_figs_x[x_val]].x_range
        fig.y_range = self.fig_list[ref_figs_y[y_val]].y_range


def create_widgets_1(self):
    '''Create and return a list of the widgets for tab 'Scatters'. There are
    two types of widgets:
        1. CheckboxButtonGroup: Used to toggle the category filters
        2. Select: A dropdown menu used to define the current colour category

    Args:
        self (Dataexplorer) : The object containing all the session information

    Returns:
        None

    '''
    cbg_list = []
    div_list = []
    for classif in self.classifs_active:
        classes = self.classes_dict[classif]  # Classes in a classification
        active_list = list(range(0, len(classes)))  # All classes start active
        cbg = CheckboxButtonGroup(labels=classes, active=active_list,
                                  width=self.window_width-(250+300+50))
        cbg_list.append(cbg)

        # Make the annotation for the CheckboxButtonGroup:
        div = Div(text='''<div style="text-align:right; font-size:12pt;
                  border:6px solid transparent">'''+classif+''':</div>''',
                  width=250)
        div_list.append(div)

    for i, cbg in enumerate(cbg_list):
        # We need the update_filter function to know who calls it, so we use
        # the "partial" function to transport that information
        cbg.on_click(partial(update_filters, caller=i, DatEx=self))

    sel = Select(title='Classification used for legend:', width=300,
                 value=self.colour_classif, options=self.classifs_active)
    sel.on_change('value', partial(update_colour_classif, DatEx=self))

    # Prepare the layout of the widgets:
    # Create rows with pairs of Div() and CheckboxButtonGroup(), where the
    # Div() contains the title. A list of those rows is combined with a
    # widgetbox of the Select widget.
    row_list = zip(div_list, cbg_list)
    div_and_cgb_cols = []
    for row_new in row_list:
        div_and_cgb_cols.append(row(*row_new))

    self.wb_list_1 = [widgetbox(sel), column(div_and_cgb_cols)]

    return


def create_widgets_2(self):
    '''Create and store a list of the widgets for tab 'Settings'. There are
    several types of widgets:
        - Button: Used to load a new file and then rebuild the whole layout
        - Div: Print text
        - RadioGroup: Single choice selection
        - CheckboxGroup: Multiple choice selection
        - Slider: Input number values in a predefined range
        - Toggle: A button to toggle two settings

    All widgets need an associated callback function that is triggered when
    the state of the widget changes. These callback functions get access to the
    Dataexplorer object and manipulate its properties. Some changes have an
    instantaneous effect (e.g. size of scatter points), if they do not have
    a huge performance impact. Most changes, however, are only triggered
    when the user switches the tabs (and thereby applies those changes).
    For these cases, the callback function set various '*_needs_update' flags.

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''

    # Button: Upload new file
    save_path = os.path.join(os.path.dirname(__file__), 'upload')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    but_load_new = new_upload_button(save_path, load_file, self,
                                     label='Upload a new file to the server')

    # Button: Download the current data as a file
    but_download = new_download_button(self)

    # Button: Export the figures as image files
    if self.server_mode:  # Not available in server mode
        but_export_png = Div(text='''<div> </div>''', height=1, width=1)
        but_export_svg = Div(text='''<div> </div>''', height=1, width=1)
    else:  # Only available locally
        but_export_png = Button(label='Export .png plots',
                                button_type='success', width=140)
        but_export_png.on_click(partial(export_figs, DatEx=self, ftype='.png'))
        but_export_svg = Button(label='Export .svg plots',
                                button_type='success', width=140)
        but_export_svg.on_click(partial(export_figs, DatEx=self, ftype='.svg'))

    # RadioGroup: Replace or append current data with new file
    rg_load = RadioGroup(labels=['Replace current data with new file',
                                 'Append new file to current data'],
                         active=self.load_mode_append)
    rg_load.on_click(partial(update_load_mode, DatEx=self))

    # Toggle: Switch coordinate center lines on and off
    tgl_coords = Toggle(label='Toggle coordinate center lines', active=True)
    tgl_coords.on_click(partial(update_coords, DatEx=self))

    # Sliders: Sliders for various settings
    sl_c_size = Slider(start=1, end=20, step=1, value=self.circle_size,
                       callback_policy='mouseup',  # Not working in 0.12.13
                       title='Set the size of the scatter points')
    sl_c_size.on_change('value', partial(update_c_size, DatEx=self))

    sl_p_h = Slider(start=100, end=1000, step=10, value=self.p_h,
                    title='Set the plot height in pixels')
    sl_p_h.on_change('value', partial(update_p_h, DatEx=self))

    sl_p_w = Slider(start=100, end=1000, step=10, value=self.p_w,
                    title='Set the plot width in pixels')
    sl_p_w.on_change('value', partial(update_p_w, DatEx=self))

    sl_vals_max = Slider(start=1, end=len(self.vals), step=1,
                         value=min(self.vals_max, len(self.vals)),
                         title='Set the maximum number of value columns')
    sl_vals_max.on_change('value', partial(update_vals_max, DatEx=self))

    # RadioGroup: Single choice list for combinator generator
    rg_comb = RadioGroup(active=self.combinator, labels=['']*3, width=600)
    rg_comb.on_change('active', partial(update_combinator, DatEx=self))
    self.rg_comb = rg_comb
    update_rg_comb(self)  # Set the actual label text of rg_comb

    # Toggle: Select all or no value columns
    tgl_all_vals = Toggle(label='Toggle all value columns', active=True)
    tgl_all_vals.on_click(partial(update_tgl_all_vals, DatEx=self))

    # CheckboxGroup: Two multiple choice selections for the used value columns
    # and classifications. The groups are wrapped in scrollable columns.
    scroll_height = max(self.window_height-410, 415)
    div_vals = Div(text='''<div style="position:relative; top:9px">
                   Select the value columns used in the plots:
                   </div>''', height=15, width=600)
    active_list = list(range(0, min(len(self.vals), self.vals_max)))
    cg_vals = CheckboxGroup(labels=self.vals, active=active_list)
    cg_vals.on_change('active', partial(update_vals_active, DatEx=self))
    cg_vals_col = column(cg_vals, sizing_mode='fixed', width=600,
                         height=scroll_height, css_classes=['scrollable'])
    self.cg_vals = cg_vals
    self.cg_vals_col = cg_vals_col

    div_classifs = Div(text='''<div style="position:relative; top:9px">
                       Select the classifications used in the plots:
                       </div>''', height=15, width=600)
    active_list = list(range(0, len(self.classifs)))
    cg_classifs = CheckboxGroup(labels=self.classifs, active=active_list)
    cg_classifs.on_change('active', partial(update_classifs_active,
                                            DatEx=self))
    cg_classifs_col = column(cg_classifs, sizing_mode='fixed', width=600,
                             height=scroll_height, css_classes=['scrollable'])
    self.cg_classifs = cg_classifs
    self.cg_classifs_col = cg_classifs_col

    # Spacer
    div_space_1 = Div(text='''<div> </div>''', height=1, width=600)  # Empty

    # Arrange the positions of widgets by listing them in the desired order
    self.wb_list_2 = [[but_load_new, rg_load, but_download,
                       but_export_png, but_export_svg],
                      div_space_1,
                      [sl_c_size, sl_p_h, sl_p_w, tgl_coords],
                      [sl_vals_max, rg_comb, tgl_all_vals],
                      [div_vals, div_classifs],
                      [cg_vals_col, cg_classifs_col]]

    return


def create_data_table(self):
    '''Create and return the DataTable widget that is shown in its own tab.

    Args:
        self (Dataexplorer): The object containing all the session information

    Return:
        data_table (DataTable) : Bokeh DataTable widget.

    '''
    create_data_table_columns(self)

    data_table = DataTable(source=self.source, columns=self.dt_columns,
                           width=self.window_width-50,
                           height=self.window_height-100,
                           fit_columns=False,
                           scroll_to_selection=False,
                           sortable=True,
                           reorderable=False,  # Reordering is not supported!
                           editable=False,  # Editing is not supported!
                           )
    self.data_table = data_table
    return self.data_table


def create_data_table_columns(self):
    '''Create the TableColumn objects required by create_data_table(). Calling
    it again allows an update of the column selection (after classifs_active
    or vals_active have changed).

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        dt_columns (TableColumn): The data table column objects
    '''
    self.dt_columns = []

    for column_ in self.get_columns_sorted():
        if self.df[column_].dtype == 'datetime64[ns]':
            dt_column = TableColumn(field=column_, title=column_,
                                    formatter=DateFormatter())
        else:
            dt_column = TableColumn(field=column_, title=column_)
        self.dt_columns.append(dt_column)

    return self.dt_columns


def create_corr_matrix_heatmap(self):
    '''Create a matrix plot containing the correlation coefficients of the
    active value columns. The applied filters are taken into account.

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        corr_matrix_heatmap (figure): A Bokeh figure with a rectangle plot
    '''
    # Apply the filters
    df_filtered = self.df[self.filter_combined_ri][self.vals_active]
    # Convert the datetime column to seconds to allow calculating correlations
    if self.col_time is not None and self.col_time in self.vals_active:
        # This only works if there is a time column and it is selected
        df_filtered[self.col_time] = pd.to_timedelta(df_filtered[self.col_time]
                                                     ).astype('timedelta64[s]')
    # Calculate the correlation matrix
    corr_matrix = df_filtered.corr(method='pearson')
    # Export the correlation matrix to Excel
    if self.export_corr_matrix:
        path = os.path.join(os.path.dirname(__file__), 'export',
                            'corr_matrix.xlsx')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        try:
            corr_matrix.to_excel(path, sheet_name=self.data_name[:30])
        except Exception as ex:
            logging.error(str(ex))
    # Call a custom function to create a heatmap figure
    self.corr_matrix_heatmap = create_heatmap(corr_matrix, self)
    # Update the selection immediately
    update_hm_source_selected(self, force=True)
    # Whenever the selection on the matrix changes, this callback happens
    self.hm_source.on_change('selected', partial(callback_heatmap, DatEx=self))

    return self.corr_matrix_heatmap


def create_info_tab():
    '''Deliver the information text for the "Info" tab.
    Returns a widgetbox containing a html div created from the contents
    of the "info.html" template file.

    Args:
        None

    Returns:
        div (widgetbox): A widgetbox containing the info text html div
    '''
    f_path = os.path.join(os.path.dirname(__file__), 'templates', 'info.html')
    with open(f_path, 'r') as f:
        info_text = f.read()

    div = Div(text=info_text, width=800)
    return widgetbox(div)


def create_layout(self):
    '''Create Bokeh 'layouts' from the widgetboxes and grid of figures and the
    content for the other tabs. The layouts are organized into tabs and those
    are added as 'root' to the current Bokeh document.

    Can be called again to remove the previous document root and add the new
    one after new data has been loaded.
    The 'tabs' widget is associated with a callback function that allows
    triggering events when the user switches between the tabs (used to update
    their content).

    Args:
        self (Dataexplorer): The object containing all the session information

    Return:
        None
    '''
    layout_1 = layout([self.wb_list_1, self.legend_top, self.grid])
    layout_2 = layout(self.data_table)
    layout_3 = layout(self.corr_matrix_heatmap)
    layout_4 = layout(self.wb_list_2)
    layout_5 = layout(create_info_tab())

    tab_1 = Panel(child=layout_1, title='Scatters')
    tab_2 = Panel(child=layout_2, title='Data Table')
    tab_3 = Panel(child=layout_3, title='Correlation')
    tab_4 = Panel(child=layout_4, title='Settings')
    tab_5 = Panel(child=layout_5, title='Info')
    tabs = Tabs(tabs=[tab_1, tab_2, tab_3, tab_4, tab_5])
    tabs.on_change('active', partial(callback_tabs, DatEx=self))

    enable_responsiveness(self, tabs)  # enable resizing of some UI elements

    curdoc().clear()  # Clear any previous document roots
    curdoc().add_root(tabs)  # Add a new root to the document
    curdoc().title = 'DataExplorer: '+self.data_name  # Set browser title

    # Things to when not in a debugging mode:
    if self.server_mode is True:
        # Activate 'leave page' confirmation dialog after a certain time
        curdoc().add_timeout_callback(partial(update_nav_confirm, active=True),
                                      timeout_milliseconds=2*60*1000)

#    table_old = curdoc().roots[0].tabs[1].child
#    print(table_old)


def prepare_filter(self):
    '''Prepare the filters used to explore the data. A filter is a Pandas
    series with the same index as the main DataFrame and booleans as values.
    Each classification needs its own filter for its classes. The filters start
    all 'True' and will be modified later, based on the user input.

    The result is filter_list, a list containing all the filters, one for each
    classification.

    Args:
        self (Dataexplorer): The object containing all the session information

    Returns:
        None

    '''
    filter_list = []  # List of Pandas series
    filter_true = []  # Define here, overwrite below, so we can use again later
    for classif in self.classifs:
        classes = self.classes_dict[classif]
        filter_true = self.df[classif].isin(classes)  # Pandas series
        filter_list.append(filter_true)  # List of Pandas series

    self.filter_list = filter_list
    self.filter_true = filter_true
    self.filter_combined = filter_true
    self.filter_combined_ri = filter_true  # Re-indexed filter_combined
    return


def update_filters(active, caller, DatEx):
    '''Function associated with the CheckboxButtonGroups (CBG). Each CBG has
    one corresponding filter (which belongs to one classification). The calling
    CBG identifies itself with the "caller" argument. It delivers a list of the
    positions of the buttons which are now active (after the user input). The
    positions are translated into classes strings (which are the button
    labels). The filters are updated based on the selected classes and then
    the DataFrame is filtered accordingly. Updating Bokeh's source object makes
    all the figures update, as well.

    Args:
        active (List) : A list of the positions of those buttons, which are
            currently active.

        caller (int) : Number of the CBG which is calling this function

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None

    '''
    i = caller  # Index of classification in classifs_active
    classif_sel = DatEx.classifs_active[i]  # Selected Classification
    classes = DatEx.classes_dict[classif_sel]  # Categories within that label

    # Translate the active button positions into chosen category strings
    classes_active = []
    for j in active:
        classes_active.append(classes[j])

    # Get a boolean filter of the selected category label, where the selected
    # categories are "True". Store this in the correct filter_list
    DatEx.filter_list[i] = DatEx.df[classif_sel].isin(classes_active)

    # "Multiply" all filters to get one combined filter (Booleans are compared
    # with "&"). We start with all entries "True". Then we compare all filters
    # in the filter_list. In the end, only those rows remain "True" which are
    # "True" in all filters.
    filter_combined = DatEx.filter_true
    for filter_i in DatEx.filter_list:
        filter_combined = filter_combined & filter_i
    DatEx.filter_combined = filter_combined

    # Selections would get messed up after the filtering.
    # Force 'empty' selection of rows = deselect everything.
    DatEx.source.selected.indices = []

    update_CDS(DatEx)  # Update ColumnDataSource to apply the filter_combined

    # The 'view' function seemed useful, but may not be flexible enough:
    # - Filtering one "column_name" for multiple "group"s seems not possible
    # - Changing the view did not seem to affect the DataTable
#    view.filters = [GroupFilter(column_name='Category Label 1', group='A'),
#                    GroupFilter(column_name='Category Label 2', group='First')
#                    ]
#    view.filters = [BooleanFilter(filter_combined)]

    # The correlation matrix now needs an update
    DatEx.corr_matrix_needs_update = True


def update_CDS(DatEx):
    '''Update the ColumnDataSource object from the Pandas DataFrame while
    applying the 'filter_combined'.

    Due to the sorting of DataFrame df in update_colours(), the user's
    selection of rows will become corrupted when we update the ColumnDataSource
    from the differently indexed DataFrame df. This is corrected by finding the
    selected rows from the old index in the new index and updating the
    selection accordingly.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    # The order of df may have changed due to sorting by the colour_classif.
    # Thus the order of filter_combined and the df have to be matched
    DatEx.filter_combined_ri = DatEx.filter_combined.reindex(DatEx.df.index)

    # Here we correct the row selection. First we get the current selection
#    source_index_sel_old = DatEx.source.selected['1d']['indices']
    source_index_sel_old = DatEx.source.selected.indices

    if source_index_sel_old == []:
        source_index_sel_new = []  # No selection found, skip the rest
    else:
        # Get indices of old selection in index of previous df
        df_index_sel_old = DatEx.df_index_last[source_index_sel_old].values
        # Get index of current df
        df_index = list(DatEx.df[DatEx.filter_combined_ri].index.values)
        # Find the new selection by matching the indices
        source_index_sel_new = [df_index.index(i) for i in df_index_sel_old]

        # Make selection empty (All glyphs vanish for a moment. Otherwise
        # on updating 'source' the wrong glyphs are selected for a moment.)
        DatEx.source.selected.indices = []

    # Update the 'data' property of the 'source' object with the new data.
    # The new data is formatted with Pandas as a dict of the 'list' type.
    DatEx.source.data = DatEx.df[DatEx.filter_combined_ri].to_dict('list')
    # Bokeh detects changes to the 'source' automatically and updates the
    # figures, glyphs and DataTable accordingly.

    # Set new row selection
    if source_index_sel_new != []:
        DatEx.source.selected.indices = source_index_sel_new

    '''After this step the script is idle until the next user input occurs.'''

    # ... unless we are in DataShader render mode:
    if DatEx.render_mode == 'DataShader':
        for i, x_y_vals in enumerate(DatEx.selected_figs):
            x_val, y_val = x_y_vals

            update_datashader(DatEx, i, x_val, y_val)


def update_colour_classif(attr, old, new, DatEx):
    '''Function associated with drop-down menu widget for the classification
    used for the legend.
    The selected classification becomes the new 'colour_classif', then the
    correct updates have to be applied.

    Args:
        attr (str): Calling widget's updated attribute

        old (str): Previous user selection

        new (str): Selected classification name

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    DatEx.colour_classif = new  # Save the new colour_classif
    update_colours(DatEx)  # Update the DataFrame
    update_CDS(DatEx)  # Update the ColumnDataSource


def update_colours(DatEx):
    '''Update (or create) the columns 'Legend' and 'Colours' of the DataFrame.
    The column 'Colours' contains the hex values of the colours of each row.
    All figures reference this column as their source for the glyph colours.
    The column 'Legend' contains the class names of the classification used
    for the legend. The legend figures reference this column as their source
    for the legend text.
    The DataFrame is sorted by the contents of the colour_classif column to
    make the legend figures display properly sorted.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Return:
        None

    '''
    try:  # Store index before sorting
        DatEx.df_index_last = DatEx.df[DatEx.filter_combined_ri].index
    except Exception:  # filter_combined_ri may not exist yet
        DatEx.df_index_last = DatEx.df.index

    # Prepare for sorting not only the colour_classif classes, but also the
    # remaining classification columns to avoid unsorted data tables
    rest = DatEx.classifs_active.copy()
    rest.remove(DatEx.colour_classif)
    sort_list = [DatEx.colour_classif] + rest  # classifs_active list reordered

    # Now sort df and update/create columns
    colormap = get_colourmap(DatEx, DatEx.classes_dict[DatEx.colour_classif])
    DatEx.df.sort_values(by=sort_list, inplace=True)
    DatEx.df['Legend'] = DatEx.df[DatEx.colour_classif]
    DatEx.df['Colours'] = [colormap[x] for x in DatEx.df[DatEx.colour_classif]]
    DatEx.colormap = colormap
    return


def update_rg_comb(DatEx):
    '''Update the labels of the RadioGroup rg_comb (the single choice selector
    for the combinatoric generator) with the new number of plots.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    DatEx.rg_comb.active = DatEx.combinator

    combis_0 = str(len(list(itertools.combinations(DatEx.vals_active, r=2))))
    combis_1 = str(len(list(itertools.permutations(DatEx.vals_active, r=2))))
    combis_2 = str(len(list(itertools.product(DatEx.vals_active, repeat=2))))
    combis_3 = str(len(DatEx.selected_figs))

    DatEx.rg_comb.labels = [
        combis_0+' plots (No inverted plots)',
        combis_1+' plots (With inverted plots)',
        combis_2+' plots (Equivalent to full correlation coefficient matrix)',
        combis_3+' plots (Custom selection in "Correlation" tab)']

    if DatEx.col_time is not None and DatEx.col_time in DatEx.vals_active:
        # Time series option is only shown when applicable
        combis_4 = str(len(get_time_series_selections(DatEx)))
        DatEx.rg_comb.labels.append(combis_4+' plots (Show time series only)')


def update_tgl_all_vals(active, DatEx):
    '''Function associated with toggle widget 'tgl_all_vals'.
    Select all (while respecting the threshold) or no value columns in the
    CheckboxGroup cg_vals.

    Args:
        active (bool): State of the toggle button

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    if active is True:
        DatEx.cg_vals.active = list(range(0, min(len(DatEx.vals),
                                                 DatEx.vals_max)))
    else:
        DatEx.cg_vals.active = []


def update_vals_active(attr, old, new, DatEx):
    '''Function associated with value column selection widget 'cg_vals'.
    Updates the vals_active (setting of active value columns), or refuses the
    update if the threshold is crossed. Setting the *_needs_update flags will
    apply the changes when the user switches the tabs.

    Args:
        attr (str): Calling widget's updated attribute

        old (List): Previous user selection

        new (List): Selected value columns (as list of button positions)

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''

    if len(new) > DatEx.vals_max:
        message = 'Maximum of '+str(DatEx.vals_max)+' value columns exceeded.'
        show_info(message)
        if not len(old) > DatEx.vals_max:  # Prevent infinite loop
            DatEx.cg_vals.active = old  # Reset selection to old state
        return
    else:
        # Translate the active button positions into chosen category strings:
        DatEx.vals_active = [DatEx.vals[j] for j in new]

        # Check for no longer valid selections:
        if DatEx.combinator == 3 or \
           (DatEx.combinator == 4 and DatEx.col_time not in DatEx.vals_active):
            # Reset the figure selection, because it is no longer valid
            DatEx.combinator = 0
            DatEx.selected_figs = get_combinations(DatEx)

        # Update labels of figure selection RadioGroup
        update_rg_comb(DatEx)

        # Set the required update flags:
        DatEx.grid_needs_update = True
        DatEx.table_needs_update = True
        DatEx.corr_matrix_needs_update = True


def update_classifs_active(attr, old, new, DatEx):
    '''Function associated with classification selection widget 'cg_classifs'.
    Updates the classifs_active (setting of active classifications), or refuses
    the update if the threshold is crossed. Setting the *_needs_update flags
    will apply the changes when the user switches the tabs.

    Args:
        attr (str): Calling widget's updated attribute

        old (List): Previous user selection

        new (List): Selected classifications (as list of button positions)

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''

    if len(new) == 0:
        message = 'Please select at least one classification.'
        show_info(message)
        DatEx.cg_classifs.active = old  # Reset selection to old state
        return
    else:
        # Translate the active button positions into chosen category strings:
        DatEx.classifs_active = [DatEx.classifs[j] for j in new]
        # If necessary, choose new default colour classification
        if DatEx.colour_classif not in DatEx.classifs_active:
            DatEx.colour_classif = DatEx.classifs_active[0]
        # Set update required flags
        DatEx.table_needs_update = True
        DatEx.classifs_need_update = True


def update_coords(active, DatEx):
    '''Function associated with toggle widget 'tgl_coords'.
    Directly updates the 'visible' attribute of the spans (which are the lines
    marking the center of the coordinate system).

    Args:
        active (bool): State of the toggle button

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    for span in DatEx.span_list:
        span.visible = active


def update_load_mode(active, DatEx):
    '''Function associated with single choice selection widget used to choose
    between replacing or appending data on the next upload 'rg_load'.
    Directly updates the 'load_mode_append' parameter.

    Args:
        active (int): State of the selection (0 or 1)

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    DatEx.load_mode_append = active


def update_c_size(attr, old, new, DatEx):
    '''Function associated with the scatter point size slider widget sl_c_size.
    Updates the size property of all GlyphRenderers (i.e. the scatter points).

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int): Selected size

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    DatEx.circle_size = new
    for glpyh_renderer in DatEx.glyph_list:
        glpyh_renderer.glyph.size = DatEx.circle_size


def update_p_h(attr, old, new, DatEx):
    '''Function associated with the plot height slider widget 'sl_p_h'. Updates
    the 'p_h' property. Change is applied on the next tab switch.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int): Selected size

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    DatEx.p_h = new
    DatEx.grid_needs_update = True


def update_p_w(attr, old, new, DatEx):
    '''Function associated with the plot width slider widget 'sl_p_w'. Updates
    the 'p_w' property. Change is applied on the next tab switch.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int): Selected size

        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    DatEx.p_w = new
    DatEx.grid_needs_update = True


def update_vals_max(attr, old, new, DatEx):
    '''This function is triggered by the 'sl_vals_max' slider widget.
    The user input value 'new' is stored in the global variable vals_max.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int) : User input

    Return:
        None
    '''
    DatEx.vals_max = new


def update_combinator(attr, old, new, DatEx):
    '''This function is triggered by the 'rg_comb' RadioGroup widget.
    The user input value 'new' is stored in the global variable combinator.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int) : User input

    Return:
        None
    '''
    DatEx.combinator = new
    DatEx.grid_needs_update = True


def update_hm_source_selected(DatEx, force=False):
    '''Update the selection of the heatmap source. We create a Pandas Series
    from the full list of possible selections. We filter it with the currently
    selected figures. The result is a list of the indices that we can pass to
    hm_source.selected.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

        force (bool, optional): Force update of selection

    Returns:
        None
    '''
    figs_all = list(itertools.product(DatEx.vals_active, repeat=2))
    figs_all_s = pd.Series(range(len(figs_all)), index=figs_all)
    figs_sel = get_combinations(DatEx)
    hm_sel_ids = figs_all_s[figs_sel].tolist()
    if hm_sel_ids != DatEx.hm_sel_ids or force is True:  # Is update necessary?
        DatEx.hm_sel_ids = hm_sel_ids
        DatEx.hm_source.selected = Selection(indices=DatEx.hm_sel_ids)


def callback_tabs(attr, old, new, DatEx):
    '''This function is triggered by selecting any of the tabs. Depending on
    the selected tab and the state of the *_needs_update flags, updates to
    the tab's contents are triggered.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int) : Number of selected tab

        DatEx (Dataexplorer): The object containing all the session information

    Return:
        None
    '''
    if new == 0:  # First tab
        if DatEx.grid_needs_update:
            toggle_loading_mouse(True)  # Begin loading mouse animation
            update_gridplot(DatEx)
            DatEx.grid_needs_update = False
            toggle_loading_mouse(False)  # End loading mouse animation

        if DatEx.classifs_need_update:
            toggle_loading_mouse(True)  # Begin loading mouse animation
            update_classifs(DatEx)
            DatEx.classifs_need_update = False
            toggle_loading_mouse(False)  # End loading mouse animation

    elif new == 1:  # Second tab
        if (DatEx.table_needs_update):
            toggle_loading_mouse(True)  # Begin loading mouse animation
            update_table(DatEx)
            DatEx.table_needs_update = False
            toggle_loading_mouse(False)  # End loading mouse animation

    elif new == 2:  # Third tab
        if (DatEx.corr_matrix_needs_update):
            toggle_loading_mouse(True)  # Begin loading mouse animation
            update_corr_matrix_heatmap(DatEx)
            DatEx.corr_matrix_needs_update = False
            toggle_loading_mouse(False)  # End loading mouse animation

        update_hm_source_selected(DatEx)

    elif new == 3:  # Fourth tab
        update_rg_comb(DatEx)

        # Update the callback of the existing download_button with a new one
        callback_updated = new_download_button(DatEx).callback
        curdoc().set_select(selector={'name': 'download_button'},
                            updates={'callback': callback_updated})


def callback_heatmap(attr, old, new, DatEx):
    '''This function is triggered when the selection in the correlation
    coefficient matrix heatmap changes.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int) : Updated selection

        DatEx (Dataexplorer): The object containing all the session information

    Return:
        None
    '''
    DatEx.hm_sel_ids = new.indices
    figs_all = pd.Series(list(itertools.product(DatEx.vals_active, repeat=2)))
    if DatEx.hm_sel_ids == []:  # Happens when clicking the reset button
        DatEx.selected_figs = figs_all
        DatEx.combinator = 2  # Equals this combinator selection
    else:
        DatEx.selected_figs = figs_all[DatEx.hm_sel_ids]

        # The combinator only needs to be updated if the selection does not
        # match the current default selection scheme.
        if DatEx.selected_figs.tolist() != get_combinations(DatEx):
            DatEx.combinator = 3

    # Set the required update flags:
    DatEx.grid_needs_update = True


def update_gridplot(DatEx):
    '''Update the gridplot in the 'Scatters' tab.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    # Get the layout from current document:
    layout_1 = curdoc().roots[0].tabs[0].child

    # For removing items, the children of a layout can be treated like a list:
    layout_1.children.remove(DatEx.grid)
#    layout_1.children.remove(DatEx.legend_top)

    # Create a new grid:
    create_plots(DatEx)  # This updates DatEx.grid
#    layout_1.children.append(DatEx.legend_top)  # Currently causes error
    layout_1.children.append(DatEx.grid)


def update_classifs(DatEx):
    '''Update the classifications in the 'Scatters' tab.
    Recreate the widgets on the first tab and replace the old widgets.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    create_widgets_1(DatEx)  # Create the widgets on tab_1 from scratch
    layout_1 = curdoc().roots[0].tabs[0].child  # Locate layout_1
    layout_1.children[0].children = DatEx.wb_list_1  # Replace the old widgets

    # Reset all the filters that might have been chosen
    prepare_filter(DatEx)
    # Update all plots with the new filters and the current colour selection
    update_colour_classif(0, 0, DatEx.colour_classif, DatEx)


def update_table(DatEx):
    '''Update the columns in the 'Data Table' tab.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    create_data_table_columns(DatEx)
    DatEx.data_table.columns = DatEx.dt_columns


def update_corr_matrix_heatmap(DatEx):
    '''Update the correlation matrix figure in the 'Correlation' tab.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    layout_3 = curdoc().roots[0].tabs[2].child
    # The children of a layout can be treated like a list:
    layout_3.children.remove(DatEx.corr_matrix_heatmap)
    layout_3.children.append(create_corr_matrix_heatmap(DatEx))


def get_palette_default():
    '''The default colour palette is Bokeh's "Category20" with one change:
    The regular colours are made more distinguishable by first using the even,
    then the odd numbers in the predefined colour palette.
    '''
    palette_old = palettes.all_palettes['Category20'][20]
    palette_new = palettes.all_palettes['Category20'][20]
    for i, colour in enumerate(palette_old):
        if i < 10:
            j = 2*i  # Even numbers
        else:
            j = 2*(i-9)-1  # Odd numbers
        palette_new[i] = palette_old[j]  # overwrite palette_new

    return palette_new


def get_colourmap(DatEx, classes):
    '''This function creates a dictionary of classes and their colours.
    It uses the list of colors ``DatEx.palette``, which was either defined by
    ``get_palette_default()`` or by the user configuration yaml.
    If the list of classes is longer than the palette, we fall back to a
    large palette with 256 continuous colors. This can be defined in the
    config, too. Selectable palette names can be found here:

    https://bokeh.pydata.org/en/latest/docs/reference/palettes.html#large-palettes

    If all fails, the function handles the possible exception thrown when
    the palette is not long enough by appending the colour grey.

    Args:
        classes (List) : List of classes.

    Return:
        colourmap (Dict) : Dictionary of classes and their colours.
    '''
    if len(classes) > len(DatEx.palette) and DatEx.palette_large_name:
        # If necessary, try to use a large palette
        try:
            DatEx.palette = getattr(
                    bokeh.palettes, DatEx.palette_large_name)(len(classes))
        except Exception as ex:
            logging.error('There is a problem with the palette "'
                          + DatEx.palette_large_name + '"...')
            logging.exception(ex)

    # Map a color to each class
    colourmap = dict()
    for i, class_ in enumerate(classes):
        try:
            colourmap[class_] = DatEx.palette[i]
        except Exception:
            colourmap[class_] = 'grey'

        # Individual classes can be overwritten by the YAML config
        if class_ in DatEx.colourmap_user:
            colourmap[class_] = DatEx.colourmap_user[class_]

    return colourmap


def get_combinations(DatEx):
    '''Return the current possible combinations of selectable plots,
    dependent of the combinator and the current active values.

    Args:
        DatEx (Dataexplorer): The object containing all the information

    Returns:
        combis (list): List of pairs of value column names
    '''
    # A choice of combinatoric generators with increasing number of results:
    if DatEx.combinator == 0:
        combis = itertools.combinations(DatEx.vals_active, r=2)
    elif DatEx.combinator == 1:
        combis = itertools.permutations(DatEx.vals_active, r=2)
    elif DatEx.combinator == 2:
        combis = itertools.product(DatEx.vals_active, repeat=2)
    elif DatEx.combinator == 3:
        combis = DatEx.selected_figs
    elif DatEx.combinator == 4:  # Time series only
        combis = get_time_series_selections(DatEx)

    return list(combis)


def get_time_series_selections(DatEx):
    '''Construct plot selection list that only contains the time series plots.

    Args:
        DatEx (Dataexplorer): The object containing all the information

    Returns:
        time_series_sel (list): List of pairs of value column names
    '''
    vals_list = DatEx.vals_active.copy()
    try:
        vals_list.remove(DatEx.col_time)
    except ValueError:  # Will throw if col_time is not in the list
        pass

    time_series_sel = [(DatEx.col_time, val) for val in vals_list]
    return time_series_sel


def load_file(filepath, DatEx):
    '''The chosen file is read into a Pandas DataFrame and the UI is recreated.
    Supported file types are '.xlsx' and '.xls'. Pandas will also try to read
    in '.csv' files, but can easily fail if the separators are not guessed
    correctly. Support for special formats of .csv files can be implemented
    in read_csv_formats().
    In order to regenerate all widgets and figures, Dataexplorer() is called to
    create a new object. The initialization of that object finishes with
    calling create_layout(), which 'clears' the current Bokeh document and
    adds a new root to the empty document.

    Args:
        filepath (str) : The path to the file to load.

        DatEx (Dataexplorer): The object containing all the session information

    Return:
        None
    '''
    if len(filepath) == 0:  # No file selected, or file dialog canceled
        return  # Return, instead of completing the function

    logging.info('Trying to open file: ' + filepath)
    try:
        df_new = read_filetypes(filepath)
    except Exception as ex:
        # Show the error message in the terminal and in a pop-up message box:
        show_info('Error: File ' + os.path.basename(filepath) +
                  ' not loaded. ' + str(ex))
        return  # Return, instead of completing the function

    logging.debug('Loaded ' + filepath)

    if DatEx.server_mode is True:
        logging.debug('Removing ' + filepath)
        os.remove(filepath)  # Remove the file after loading its data

    '''Now that the new data is loaded, we need to replace the old data or
    append to it'''
    if not DatEx.load_mode_append:  # Means: Load mode = replace
        data_name = os.path.basename(filepath)
        df = df_new
    else:  # Means: Load mode = append
        # Appended DataFrames get their names as a classification 'File'
        data_name_new = os.path.basename(filepath)
        data_name = DatEx.data_name+', '+data_name_new
        df_new['File'] = [data_name_new]*len(df_new)
        df_old = DatEx.df.drop(columns=['Colours', 'Legend'])
        if 'File' not in df_old.columns:  # Only do this if necessary
            df_old['File'] = [DatEx.data_name]*len(df_old)

        # Append (with concatenate) the old and new df, with a new index
        df = pd.concat([df_old, df_new], ignore_index=True, sort=False)
        try:
            # If there is a column named 'Time', move it to the first position
            df = df.set_index('Time').reset_index()
        except Exception as ex:
            logging.debug(ex)
            pass

    '''Start the recreation of the UI by creating a new Dataexplorer object'''
    DatEx = Dataexplorer(df, data_name, DatEx.server_mode,
                         window_height=DatEx.window_height,
                         window_width=DatEx.window_width,
                         render_mode=DatEx.render_mode)
    # (The script is basically restarted at this point)


def show_info(message):
    '''Shows a notification window with the given message in the browser.
    Use the browsers 'alert()' JavaScript code that shows a pop-up.

    Args:
        message (str) : Message text.

    Return:
        None
    '''
    logging.critical(message)  # Output to log with priority 'critical'

    message = re.sub(r'\\', r'\\\\', message)  # Replace '\' with '\\'
    message = re.sub(r'\n', r'\\n', message)  # Replace 'newline' with '\\n'
    message = re.sub("'", "", message)  # Message cannot contain: '
    js_code = """alert('"""+message+"""')"""
    run_js_code(js_code)


def update_nav_confirm(active):
    '''Enable or disable the browser's 'leave page' confirmation dialog.

    Args:
        active (bool): State of the toggle button

    Returns:
        None
    '''
    if active:
        js_code = '''window.onbeforeunload = function() {
                        return true;
                        };
                  '''
    else:
        js_code = '''window.onbeforeunload = null;'''

    run_js_code(js_code)
    logging.critical('nav_confirm is now '+str(active))


def toggle_loading_mouse(active):
    '''Enable or disable a loading mouse animation.

    Args:
        active (bool): State of the toggle

    Returns:
        None
    '''
    if active:
        js_code = '''document.body.style.cursor = "wait";
                  '''
    else:
        js_code = '''document.body.style.cursor = "auto";
                     window.scroll(0, 0);
                  '''
    run_js_code(js_code)


def run_js_code(js_code):
    '''Run arbitrary JavaScript code anytime.
    A TextInput widget is used to execute the JavaScript code. This widget is
    added, used and then immediately removed from the document.

    Args:
        js_code (str) : JavaScript code to be executed.

    Return:
        None
    '''
    ti_code = TextInput(value='')
    ti_code.js_on_change('value', CustomJS(code=js_code))
    curdoc().add_root(ti_code)
    ti_code.value = ' '
    curdoc().remove_root(ti_code)


def export_figs(DatEx, fig_sel=None, ftype='.png'):
    '''Export a specific or all figures to files of a given file type.
    File type can be ``'.png'`` (raster) or ``'.svg'`` (vector).

    Due to current (Bokeh 0.13.0) limitations in the function export_png,
    the models we try to export must not be assigned to a document. So during
    the export, we have to remove the complete user interface from
    ``curdoc()``. It is replaced with a status message.

    Args:
        DatEx (Dataexplorer) : The Dataexplorer object.

        fig_sel (figure, optional) : A Bokeh figure. If ``None``, all figures
        are exported.

        ftype (str, optional) : A string defining the file type extension.
        Options: ``'.png'`` (raster) or ``'.svg'`` (vector).

    Returns:
        None
    '''
    toggle_loading_mouse(True)  # Enable spinning mouse wheel
    # Create a temporary view during the export process
    div_temp = Div(text='''<div>Please wait</div>''', width=1000)
    root_temp = layout(div_temp)
    curdoc().add_root(root_temp)
#    root_main = curdoc().roots[0]  # bokeh < 1.0.0: required for export_png
#    curdoc().remove_root(root_main)  # bokeh < 1.0.0: required for export_png

    out_folder = os.path.join(os.path.dirname(__file__), 'export')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if fig_sel is None:
        export_list = [DatEx.legend_top, DatEx.legend_bot,
                       DatEx.corr_matrix_heatmap] + DatEx.fig_list
    else:
        export_list = [DatEx.legend_top, DatEx.legend_bot,
                       DatEx.corr_matrix_heatmap] + [fig_sel]

    for i, fig in enumerate(export_list):
        # Normalize or replace invalid characters ('W/m²' becomes 'W_m2')
        fig_name = unicodedata.normalize('NFKD', fig.name)  # normalize
        fig_name = re.sub(r'[\/\:\?\!\<\>\|\*\"]', '_', fig_name)  # replace
        out_file = os.path.join(out_folder, fig_name+ftype)

        logging.info('Exporting '+out_file)
        div_temp.text = 'Please wait... exporting '+out_file
        try:
            if ftype == '.png':  # Export as raster graphic
                export_png(fig, filename=out_file)
            elif ftype == '.svg':  # Export as vector graphic
                fig.output_backend = "svg"
                export_svgs(fig, filename=out_file)
        except Exception as ex:
            logging.critical(str(ex))

    # Restore original view
#    curdoc().add_root(root_main)  # bokeh < 1.0.0: required for export_png
    curdoc().remove_root(root_temp)
    toggle_loading_mouse(False)  # Disable spinning mouse wheel


def update_datashader(DatEx, i, x_val, y_val):
    p = DatEx.fig_list[i]
    df_copy = DatEx.df[DatEx.filter_combined_ri].copy()

    # Prepare the axis ranges, converting DateTime to float
    ref = pd.datetime(1970, 1, 1, 00, 00, 00)
    if DatEx.df[x_val].dtype == 'datetime64[ns]':
        df_copy[x_val] = (df_copy[x_val] - ref)/np.timedelta64(1, 'ms')
    if DatEx.df[y_val].dtype == 'datetime64[ns]':
        df_copy[y_val] = (df_copy[y_val] - ref)/np.timedelta64(1, 'ms')

#    x_range = [df_copy[x_val].min(), df_copy[x_val].max()]
#    y_range = [df_copy[y_val].min(), df_copy[y_val].max()]

    # Contruct the DataShader image
    cvs = DatEx.cvs_list[i]
    try:
        # Will fail if plot would be empty
        agg = cvs.points(df_copy, x_val, y_val,
                         agg=ds.count_cat('Legend'))
        img = tf.shade(agg, color_key=DatEx.colormap, how='eq_hist',
                       min_alpha=165)
        img = tf.dynspread(img, threshold=0.99)
        # Replace only the image array in the plots DataSource
        p.renderers[7].data_source.data['image'] = [img.data]
    except ZeroDivisionError as ex:
        logging.error(ex)
        # Create an empty plot
        p.renderers[7].data_source.data['image'] = []
        pass
    except Exception as ex:
        logging.exception(ex)
        # Create an empty plot
        print(p.name)  # TODO remove after testing
        print(p.renderers)  # TODO remove after testing
        pass

#    p.x_range.start = x_range[0]
#    p.x_range.end = x_range[1]
#    p.y_range.start = y_range[0]
#    p.y_range.end = y_range[1]

    return p


def datashader_callback(attr, old, new, p, DatEx, x_val, y_val):
    '''Not working properly yet (is supposed to allow redrawing the image
    when zooming)
    '''
#    print(p, attr, old, new)
    df_copy = DatEx.df[DatEx.filter_combined_ri].copy()
    x_range = [p.x_range.start, p.x_range.end]
    y_range = [p.y_range.start, p.y_range.end]
    print(x_range, y_range)
    cvs = ds.Canvas(plot_width=DatEx.p_w, plot_height=DatEx.p_h,
                    x_range=x_range, y_range=y_range)
    agg = cvs.points(df_copy, x_val, y_val, agg=ds.count_cat('Legend'))
    img = tf.shade(agg, color_key=DatEx.colormap, how='eq_hist')
    img = tf.dynspread(img, threshold=0.75)
    print(img.data)
    p.renderers[7].data_source.data['image'] = [img.data]
#    p.image_rgba(image=[img.data], x=p.x_range[0], y=p.y_range[0],
#                 dw=p.x_range[1]-p.x_range[0],
#                 dh=p.y_range[1]-p.y_range[0])


if __name__ == "__main__":
    '''
    Main function for debugging purposes:

    This function is executed when the script is started directly with
    Python. We create an initial set of test data and then create the
    DataExplorer user interface. Does not produce an output.
    '''
    df = create_test_data()
    data_name = 'Example Data'
    server_mode = False

    Dataexplorer(df, data_name, server_mode)
