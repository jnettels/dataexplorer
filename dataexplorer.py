# -*- coding: utf-8 -*-
'''
@version: 0.10

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

TODO: Fix axis ranges
TODO: Highlight a plot (e.g. add red border) by clicking on it
TODO: Transfer session settings via DatEx.__dict__
TODO: Make combinator generator a single choice list

Known issues:
- Plots fail when Time column includes 'NaT', so those columns are removed

'''

import pandas as pd
import numpy as np
import itertools
import os
import logging
import bokeh
from bokeh.layouts import widgetbox, gridplot, layout
from bokeh.layouts import row
from bokeh.layouts import column
from bokeh.models.widgets import CheckboxButtonGroup, Select, CheckboxGroup
from bokeh.models.widgets import Div, DataTable, TableColumn, DateFormatter
from bokeh.models.widgets import Panel, Tabs, TextInput, Slider, Toggle
from bokeh.models.widgets import RadioGroup
from bokeh.models import ColumnDataSource  # , CategoricalColorMapper
from bokeh.models import CustomJS, HoverTool, Span
# from bokeh.models import CDSView, BooleanFilter, GroupFilter
from bokeh.plotting import figure
from bokeh import palettes
from bokeh.io import curdoc
from functools import partial
from pandas.api.types import is_categorical_dtype
# from pandas.api.types import CategoricalDtype
from bokeh.io import export_png, export_svgs
from distutils.version import StrictVersion

# My own library of functions from the file helpers.py
from helpers import (new_upload_button, create_test_data, create_heatmap,
                     read_filetypes, read_csv_formats, new_download_button)

# Global Pandas option for displaying terminal output
pd.set_option('display.expand_frame_repr', False)

# Check some version requirements
pd_v_required = '0.21.0'
if StrictVersion(pd.__version__) < StrictVersion(pd_v_required):
    logging.critical('Warning: Pandas version '+pd_v_required+' is required.' +
                     ' Your version is '+pd.__version__)
bk_v_required = '0.12.13'
if StrictVersion(bokeh.__version__) < StrictVersion(bk_v_required):
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

    def __init__(self, df, data_name, combinator=1, vals_max=6):
        '''Return a Dataexplorer object, the object containing all the session
        information. Initialize all object properties.
        Perform all the tasks necessary to create the Data Explorer user
        interface by calling the required functions.

        Args:
            df (Pandas DataFrame) : The input data we want to explore.

            data_name (str) : The filename of the current data set.

            combinator (int, optional): A identifier for combinatoric generator

            vals_max (int, optional): A threshold for number of value columns

        Returns:
            None
        '''
        self.df = df
        self.vals_max = vals_max  # Threshold for number of value columns
        self.data_name = data_name
        self.combinator = combinator  # Identifier for combinatoric generator
        self.grid_needs_update = False
        self.table_needs_update = False
        self.corr_matrix_needs_update = False
        self.classifs_need_update = False
        self.c_size = 5
        self.p_h = 250  # global setting for plot_height
        self.p_w = 250  # global setting for plot_width
        self.load_mode_append = 0  # 0 equals False equals replace

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

        # Create and get a list of the widgets for tab 1
        create_widgets_1(self)

        # Create and get a list of the widgets for tab 2
        create_widgets_2(self)

        # Create and get the DataTable for tab 3
        create_data_table(self)

        # Create a correlation coefficient matrix plot
        create_corr_matrix_heatmap(self)

        # Create a Bokeh "layout" from the widgets and grid of figures
        create_layout(self)

    def get_columns_sorted(self):
        # If a time column exists, it should be the first column in DataTable
        if self.col_time is not None:
            vals_wo_time = self.vals_active.copy()
            vals_wo_time.remove(self.col_time)
            columns = [self.col_time] + self.classifs_active + vals_wo_time
        else:
            columns = self.classifs_active + self.vals_active

        return columns


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
            classes_dict[classif] = sorted(classes)  # Sort and store them
        except Exception as ex:  # Map to strings before sorting
            df[classif] = list(map(str, df[classif]))
            classes = list(set(df[classif]))  # Get classes of classification
            classes_dict[classif] = sorted(classes)  # Sort and store them
            pass
#        print(classif, classes_dict[classif])

    if len(vals) > self.vals_max:
        vals_active = vals[:self.vals_max]  # Use a shortened list of vals
    else:
        vals_active = vals

    self.vals = vals
    self.classifs = classifs
    self.classes_dict = classes_dict
    self.vals_active = vals_active  # active value columns
    self.classifs_active = classifs  # active classification columns

    # The first classification is the default colour classification
    self.colour_classif = classifs[0]  # Name of current colour classification
    # Create (or update) 'Legend' and 'Colours' columns
    update_colours(self)
    # Create the Bokeh ColumnDataSource object from Pandas DataFrame
    self.source = ColumnDataSource(data=df)

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
                'plot_height': self.p_h, 'plot_width': self.p_w,
                'lod_factor': 1000,  # level-of-detail decimation
                # 'output_backend': 'webgl',  # Better performance
                # 'output_backend': 'svg',  # For export with SaveTool (Slow!)
                }
    glyph_set = {'color': 'Colours', 'hover_color': 'Colours',
                 'fill_alpha': 0.2, 'hover_alpha': 1,
                 'size': self.c_size}
    span_set = {'location': 0, 'line_color': 'grey',
                'line_dash': 'dashed', 'line_width': 1}

    # A choice of combinatoric generators with descending number of results:
    if self.combinator == 4:
        combis = itertools.product(self.vals_active, repeat=2)
    elif self.combinator == 3:
        combis = itertools.combinations_with_replacement(self.vals_active, r=2)
    elif self.combinator == 2:
        combis = itertools.permutations(self.vals_active, r=2)
    elif self.combinator == 1:
        combis = itertools.combinations(self.vals_active, r=2)

    self.fig_list = []  # List with the complete figures
    self.glyph_list = []  # List of GlyphRenderers
    self.span_list = []  # List of spans (coordinate center lines)

    for x_val, y_val in combis:
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

        # Create the actual circle GlyphRenderer
        cr = p.circle(x=x_val, y=y_val, source=self.source, **glyph_set)
        p.xaxis.axis_label = x_val
        p.yaxis.axis_label = y_val

        self.fig_list.append(p)
        self.glyph_list.append(cr)

        # Create HoverTool:
        hover = HoverTool(point_policy='follow_mouse',  # 'snap_to_data',
                          tooltips=tips_list,
                          renderers=[cr],  # Uses 'hover_*' options
                          formatters=formatters_dict)
        p.add_tools(hover)

        # Add horizontal and vertical lines in the center coordinates
        span_h = Span(**span_set, dimension='height', level='underlay')
        span_w = Span(**span_set, dimension='width', level='underlay')
        self.span_list.append(span_h)
        self.span_list.append(span_w)
        p.add_layout(span_h)
        p.add_layout(span_w)

    '''
    The plots are completed, now we add two figures for the legends that go to
    the top and bottom of the page. For a nice look, we remove all parts of the
    figures but the legends themselves.
    '''
    legend_top = figure(plot_height=50, plot_width=1850)
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
    self.fig_list.append(legend_bot)

    # Get the number of grid columns from the rounded square root of number of
    # figures.
    if self.combinator == 4:
        n_grid_cols = int(round(np.sqrt(len(self.fig_list)), 0))
    elif self.combinator == 2:
        n_grid_cols = int(round(np.sqrt(len(self.fig_list)), 0))-1
    else:
        n_grid_cols = int(round(np.sqrt(len(self.fig_list)))) + 1
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
    grid.children[1].height = 3*self.p_h
    grid.children[1].width = (n_grid_cols+1)*self.p_w

    self.grid = grid
    self.legend_top = legend_top
    return self.grid


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
                                  width=999)
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

    sel = Select(title='Classification used for legend:',
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

    # RadioGroup: Replace or append current data with new file
    rg_load = RadioGroup(labels=['Replace current data with new file',
                                 'Append new file to current data'],
                         active=self.load_mode_append)
    rg_load.on_click(partial(update_load_mode, DatEx=self))

    # Toggle: Switch coordinate center lines on and off
    tgl_coords = Toggle(label='Toggle coordinate center lines',
                        active=True)
    tgl_coords.on_click(partial(update_coords, DatEx=self))

    # Sliders: Sliders for various settings
    sl_c_size = Slider(start=1, end=20, step=1,
                       value=self.c_size,
                       title='Set the size of the scatter points')
    sl_c_size.on_change('value', partial(update_c_size, DatEx=self))

    sl_p_h = Slider(start=100, end=1000, step=10,
                    value=self.p_h,
                    title='Set the plot height in pixels')
    sl_p_h.on_change('value', partial(update_p_h, DatEx=self))

    sl_p_w = Slider(start=100, end=1000, step=10,
                    value=self.p_w,
                    title='Set the plot width in pixels')
    sl_p_w.on_change('value', partial(update_p_w, DatEx=self))

    sl_vals_max = Slider(start=2, end=len(self.vals), step=1,
                         value=min(self.vals_max, len(self.vals)),
                         title='Set the maximum number of value columns')
    sl_vals_max.on_change('value', partial(update_vals_max, DatEx=self))

    sl_comb = Slider(start=1, end=4, step=1, value=self.combinator,
                     title='Set complexity of the combinatoric generator')
    sl_comb.on_change('value', partial(update_combinator, DatEx=self))

    # CheckboxGroup: Two multiple choice selections for the used value columns
    # and classifications. The groups are wrapped in scrollable columns.
    div_vals = Div(text='''<div style="position:relative; top:15px">
                   Select the value columns used in the plots:
                   </div>''', height=25, width=600)
    active_list = list(range(0, min(len(self.vals), self.vals_max)))
    cg_vals = CheckboxGroup(labels=self.vals, active=active_list)
    cg_vals.on_change('active', partial(update_vals_active, DatEx=self))
    cg_vals_col = column(cg_vals, sizing_mode='fixed', height=500, width=600,
                         css_classes=['scrollable'])
    self.cg_vals = cg_vals

    div_classifs = Div(text='''<div style="position:relative; top:15px">
                       Select the classifications used in the plots:
                       </div>''', height=25, width=600)
    active_list = list(range(0, len(self.classifs)))
    cg_classifs = CheckboxGroup(labels=self.classifs, active=active_list)
    cg_classifs.on_change('active', partial(update_classifs_active,
                                            DatEx=self))
    cg_classifs_col = column(cg_classifs, sizing_mode='fixed', height=500,
                             width=600, css_classes=['scrollable'])
    self.cg_classifs = cg_classifs

    # Spacer
    div_space_1 = Div(text='''<div> </div>''', height=8, width=600)  # Empty

    # Arrange the positions of widgets by listing them in the desired order
    self.wb_list_2 = [[but_load_new, rg_load, but_download],
                      div_space_1,
                      [sl_c_size, sl_p_h, sl_p_w, tgl_coords],
                      [sl_vals_max, sl_comb],
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
                           fit_columns=True,
                           width=1850,
                           height=800,
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
    corr_matrix = self.df[self.filter_combined_ri][self.vals_active].corr(
            method='pearson')

    self.corr_matrix_heatmap = create_heatmap(corr_matrix)
    return self.corr_matrix_heatmap


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

    tab_1 = Panel(child=layout_1, title='Scatters')
    tab_2 = Panel(child=layout_2, title='Data Table')
    tab_3 = Panel(child=layout_3, title='Correlation')
    tab_4 = Panel(child=layout_4, title='Settings')
    tabs = Tabs(tabs=[tab_1, tab_2, tab_3, tab_4])
    tabs.on_change('active', partial(callback_tabs, DatEx=self))

    curdoc().clear()  # Clear any previous document roots
    curdoc().add_root(tabs)  # Add a new root to the document
    curdoc().title = 'DataExplorer: '+self.data_name  # Set browser title

    # Things to when not in a debugging mode:
    if logging.getLogger().getEffectiveLevel() == logging.CRITICAL:
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
    DatEx.source.selected = {'0d': {'glyph': None, 'indices': []},
                             '1d': {'indices': []},
                             '2d': {'indices': {}}}

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
    source_index_sel_old = DatEx.source.selected['1d']['indices']

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
        DatEx.source.selected = {'0d': {'glyph': None, 'indices': []},
                                 '1d': {'indices': [[]]},
                                 '2d': {'indices': {}}}

    # Update the 'data' property of the 'source' object with the new data.
    # The new data is formatted with Pandas as a dict of the 'list' type.
    DatEx.source.data = DatEx.df[DatEx.filter_combined_ri].to_dict('list')
    # Bokeh detects changes to the 'source' automatically and updates the
    # figures, glyphs and DataTable accordingly.

    # Set new row selection
    if source_index_sel_new != []:
        DatEx.source.selected = {'0d': {'glyph': None, 'indices': []},
                                 '1d': {'indices': source_index_sel_new},
                                 '2d': {'indices': {}}}

    '''After this step the script is idle until the next user input occurs.'''


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

    # Now sort df and update/create columns
    colormap = get_colourmap(DatEx.classes_dict[DatEx.colour_classif])
    DatEx.df.sort_values(by=[DatEx.colour_classif], inplace=True)
    DatEx.df['Legend'] = DatEx.df[DatEx.colour_classif]
    DatEx.df['Colours'] = [colormap[x] for x in DatEx.df[DatEx.colour_classif]]

    return


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
    elif len(new) < 2:
        return
    else:
        # Translate the active button positions into chosen category strings:
        DatEx.vals_active = [DatEx.vals[j] for j in new]
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
    DatEx.c_size = new
    for glpyh_renderer in DatEx.glyph_list:
        glpyh_renderer.glyph.size = DatEx.c_size


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
    '''This function is triggered by the 'sl_comb' slider widget.
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


def callback_tabs(attr, old, new, DatEx):
    '''This function is triggered by selecting any of the tabs. Depending on
    the selected tab and the state of the *_needs_update flags, updates to
    the tab's contents are triggered.

    Args:
        attr (str): Calling widget's updated attribute

        old (int): Previous user selection

        new (int) : Number of selected tab.

    Return:
        None
    '''
    if new == 0:  # First tab
        if DatEx.grid_needs_update:
            update_gridplot(DatEx)
            DatEx.grid_needs_update = False

        if DatEx.classifs_need_update:
            update_classifs(DatEx)
            DatEx.classifs_need_update = False

    elif new == 1:  # Second tab
        if (DatEx.table_needs_update):
            update_table(DatEx)
            DatEx.table_needs_update = False

    elif new == 2:  # Third tab
        if (DatEx.corr_matrix_needs_update):
            update_corr_matrix_heatmap(DatEx)
            DatEx.corr_matrix_needs_update = False

    elif new == 3:  # Fourth tab
        # Update the callback of the existing download_button with a new one
        callback_updated = new_download_button(DatEx).callback
        curdoc().set_select(selector={'name': 'download_button'},
                            updates={'callback': callback_updated})


def update_gridplot(DatEx):
    '''Update the gridplot in the 'Scatters' tab.

    Args:
        DatEx (Dataexplorer): The object containing all the session information

    Returns:
        None
    '''
    # Create a new grid:
    grid_new = create_plots(DatEx)

    # Get the old grid and the layout containing it from current document:
    grid_old = curdoc().roots[0].tabs[0].child.children[2]
#    grid_old = curdoc().get_model_by_name('plot_grid')  # Does not work
    layout_1 = curdoc().roots[0].tabs[0].child

    # The children of a layout can be treated like a list:
    layout_1.children.remove(grid_old)
    layout_1.children.append(grid_new)


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


def get_colourmap(classes):
    '''This function creates a dictionary of classes and their colours. It
    handles the possible exception thrown when the palette is not long enough
    by appending the colour grey. The regular colours are made more
    distinguishable by first using the even, then the odd numbers in the
    predefined colour palette.

    Args:
        classes (List) : List of classes.

    Return:
        colourmap (Dict) : Dictionary of classes and their colours.
    '''
    colourmap = dict()
    palette = palettes.all_palettes['Category20'][20]
#    palette = palettes.all_palettes['Spectral'][len(classes)]
#    palette = palettes.all_palettes['Spectral'][11]
    for i, class_ in enumerate(classes):
        if i < 10:
            j = 2*i  # Even numbers
        else:
            j = 2*(i-9)-1  # Odd numbers
        try:
            colourmap[class_] = palette[j]
        except Exception:
            colourmap[class_] = 'grey'
    return colourmap


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
        show_info('Error: File not loaded: '+filepath+' \n'+str(ex))
        return  # Return, instead of completing the function

    logging.debug('Loaded ' + filepath)

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
        df = pd.concat([df_old, df_new], ignore_index=True)
        try:
            # If there is a column named 'Time', move it to the first position
            df = df.set_index('Time').reset_index()
        except Exception as ex:
            logging.debug(ex)
            pass

    # Save some settings:
    combinator_last = DatEx.combinator

    '''Start the recreation of the UI by creating a new Dataexplorer object'''
    DatEx = Dataexplorer(df, data_name, combinator=combinator_last)
    # (The script is basically restarted at this point)


def show_info(message):
    '''Shows a notification window with the given message in the browser.
    Use the browsers 'alert()' JavaScript code that shows a pop-up.

    Args:
        message (str) : Message text.

    Return:
        None
    '''
    message = message.replace("'", "")  # Message cannot contain: '
    js_code = """alert('"""+message+"""')"""
    run_js_code(js_code)

    logging.critical(message)  # Output to log with priority 'critical'


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

    Args:
        DatEx (Dataexplorer) : The Dataexplorer object.

        fig_sel (figure, optional) : A Bokeh figure. If None, all figures are
            exported.

        ftype (str, optional) : A string defining the file type extension.

    Returns:
        None
    '''
    out_folder = os.path.join(os.path.dirname(__file__), 'export')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    if fig_sel is None:
        export_list = DatEx.fig_list
    else:
        export_list = [fig_sel]

    for i, fig in enumerate(export_list):
        out_file = os.path.join(out_folder, str(i)+ftype)
        if ftype == '.png':  # Export as raster graphic
            export_png(fig, filename=out_file)
        elif ftype == '.svg':  # Export as vector graphic
            fig.output_backend = "svg"
            export_svgs(fig, filename=out_file)


if __name__ == "__main__":
    '''
    Main function for debugging purposes:

    This function is executed when the script is started directly with
    Python. We create an initial set of test data and then create the
    DataExplorer user interface. Does not produce an output.
    '''
    df = create_test_data()
    data_name = 'Example Data'

    Dataexplorer(df, data_name)
