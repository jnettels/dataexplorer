# -*- coding: utf-8 -*-
'''
Created on Fri Nov  3 08:10:21 2017

@author: Joris Nettelstroth

The DataExplorer should help you explore correlations within your data. It
features a user interface that shows scatter plots of all the variables in
your data. Categories found in your data can be used to filter the views.

You can start this program with the "start_bokeh.cmd" batch file.

Alternatively, you can start it with your own command prompt in Windows:
    - Go to Start and type "cmd"
    - Select "Eingabeaufforderung"
    - Change directory to the folder containing the script by typing
      "cd path/to/folder"
    - Start a Bokeh server running this script by typing
      "bokeh serve dataexplorer.py --show"
    - Your webbrowser should open and display the interface of the program
    - Hit the button to load your own Excel file

The file excel_example.xlsx shows the required input format and gives hints
about the usage.

If you do not yet have Python and Bokeh installed, the easiest way to do that
is by downloading and installing "Anaconda" from here:
https://www.anaconda.com/download/
Its a package manager that distributes Python with data science packages.

During installation, please allow to add variables to $PATH (or do that
manually afterwards.) This allows Bokeh to be started from everywhere, which
is required for the batch file to work.

TODO: HoverTool with categories
TODO: Choose combinatoric generators
TODO: Gridplot scrollable
TODO: Fix axis ranges
TODO: Horizontal legend on top (+ vertical legend at bottom)

'''

import pandas as pd
import numpy as np
import itertools
import os
import logging
from bokeh.layouts import widgetbox, gridplot, layout
from bokeh.layouts import row
# from bokeh.layouts import column
from bokeh.models.widgets import CheckboxButtonGroup, Select, CheckboxGroup
from bokeh.models.widgets import Div, DataTable, TableColumn, DateFormatter
from bokeh.models.widgets import Panel, Tabs, TextInput, Slider
from bokeh.models import ColumnDataSource  # , CategoricalColorMapper
from bokeh.models import CustomJS
# from bokeh.models import CDSView, BooleanFilter, GroupFilter
from bokeh.plotting import figure
from bokeh import palettes
from bokeh.io import curdoc
from functools import partial
from tkinter import Tk, filedialog  # , messagebox
from pandas.api.types import is_categorical_dtype
# from pandas.api.types import CategoricalDtype

# My own library of functions from the file helpers.py
from helpers import new_upload_button, create_test_data


def create_dataexplorer_UI(df, filepath, data_name):
    '''Performs all the tasks necessary to create the Data Explorer user
    interface by calling the required functions. It is also used to recreate
    the UI when new data is loaded.

    Args:
        df (Pandas DataFrame) : The input data we want to explore.

        filepath (str) : Path to file to load.

        data_name (str) : The (file)name of the current data set.

    Returns:
        None

    '''

    # Get categories, their labels and vals (column names of values) from df
    try:
        source, cats, cats_labels, vals, colour_cat = analyse_dataframe(df)
    except Exception as ex:  # Will be thrown if the df has an incorrect format
        show_info(str(ex))
        return  # Skip the rest in case of an exception

    # Use Bokeh to plot the data in an interactive way
    grid = create_plots(source, df, vals, colour_cat)

    # Prepare the filters used to explore the data
    filter_list, filter_true = prepare_filter(cats, cats_labels, df)

    # Create and get a list of the widgets for tab 1
    wb_list_1 = create_widgets_1(cats, cats_labels, colour_cat, filter_list,
                                 filter_true, df, source)

    # Create and get a list of the widgets for tab 2
    wb_list_2 = create_widgets_2(filepath, vals, colour_cat, source, df, cats)

    # Create and get the DataTable for tab 3
    data_table = create_data_table(source, df, cats, vals)

    # Create a Bokeh "layout" from the widgets and grid of figures
    create_layout(wb_list_1, grid, wb_list_2, data_table, data_name)

    # The script ends here (but Bokeh keeps waiting for user input)


def analyse_dataframe(df):
    '''Analyse a given DataFrame to seperate the columns in categories and
    values. "Object" columns become categories, their column names are saved
    as category labels.

    Args:
        df (Pandas DataFrame) : The input data we want to explore.

    Returns:
        cats (List) : List of the column names that contain categories.

        cats_labels (Dict) : Dictionary containing the categories associated
            with each label.

        vals (List) : List of the column names that contain values.

        colour_cat (str) : Name of the current colour category label.
    '''

    columns = df.columns.values.tolist()
    cats = []
    vals = []
    for column in columns:
        # The column contains categories or values
        if df[column].dtype == object or is_categorical_dtype(df[column]):
            cats.append(column)
        else:
            vals.append(column)

    if cats == []:
        raise LookupError('No category columns found in the file! Please ' +
                          'refer to example Excel file for instructions.')
    elif vals == []:
        raise LookupError('No value columns found in the file! Please ' +
                          'refer to example Excel file for instructions.')

    cats_labels = dict()
    for cat in cats:
        try:  # Try to sort the categories
            entries = list(set(df[cat]))
            cats_labels[cat] = sorted(entries)
        except Exception as ex:  # Map to strings before sorting
            df[cat] = list(map(str, df[cat]))
            entries = list(set(df[cat]))
            cats_labels[cat] = sorted(entries)
            pass
#        print(cat, cats_labels[cat])

    # Set the colours as a column of the DataFrame:
    colour_cat = cats[0]  # The first colour category label is the default
    colourmap = get_colourmap(cats_labels[colour_cat])
    colour_list = [colourmap[x] for x in df[colour_cat]]
    df['Legend'] = df[colour_cat]
    df['Colours'] = colour_list

    source = ColumnDataSource(data=df)  # Create the ColumnDataSource object

    return source, cats, cats_labels, vals, colour_cat


def create_plots(source, df, vals, colour_cat):
    '''Use Bokeh to plot the data in an interactive way. The Bokeh settings
    are defined in this function, as well as the combinatoric generator which
    generates the combinations of all the "vals". For each combination, a
    figure is created and all of those figures are combined into one grid.

    Args:
        source (ColumnDataSource) : Bokeh's data format.

        df (Pandas DataFrame) : The input data we want to explore.

        vals (List) : List of the column names that contain values.

        colour_cat (str) : Name of the current colour category label.

    Returns:
        grid (Gridplot) : Grid containing all created figures.

    '''

    if len(vals) > vals_max:
        vals = vals[:vals_max]

#    view = CDSView(source=source, filters=[])  # Create an empty view object

    plot_size_and_tools = {'tools': ['pan', 'wheel_zoom', 'box_zoom', 'reset',
                                     'lasso_select', 'box_select', 'save',
                                     'help'],
                           'active_scroll': 'wheel_zoom',
                           'plot_height': 250, 'plot_width': 250,
                           'lod_factor': 1000,  # level-of-detail decimation
                           }

#    bokeh_colormap = CategoricalColorMapper(palette=palette,
#                                            factors=cats_labels[colour_cat])
#    colour_def = {'field': colour_cat, 'transform': bokeh_colormap}

    # A choice of combinatoric generators with decending number of results:
#    permutation_list = itertools.product(vals, repeat=2)
#    permutation_list = itertools.permutations(vals, r=2)
#    permutation_list = itertools.combinations_with_replacement(vals, r=2)
    permutation_list = itertools.combinations(vals, r=2)

    fig_list = []  # List with the complete figures
    glyph_list = []  # List with the glyphs contained in the figures
    for permutation in permutation_list:
        x_val = permutation[0]
        y_val = permutation[1]
        if df[x_val].dtype == 'datetime64[ns]':
            p = figure(**plot_size_and_tools, x_axis_type='datetime')
        elif df[y_val].dtype == 'datetime64[ns]':
            p = figure(**plot_size_and_tools, y_axis_type='datetime')
        else:
            p = figure(**plot_size_and_tools)

        glyph = p.circle(x=x_val, y=y_val, source=source,
                         legend='Legend',
                         color='Colours',
                         fill_alpha=0.2,
                         size=5
                         )
        p.xaxis.axis_label = x_val
        p.yaxis.axis_label = y_val
        p.legend.visible = False
        p.legend.location = 'top_left'
        p.legend.click_policy = 'hide'
        fig_list.append(p)
        glyph_list.append(glyph)

    '''
    The plots are completed, now we add a figure for the legend. Here we remove
    everything but the legend itself. This figure is last in the grid.
    '''
    legend_fig = figure(plot_height=500, plot_width=500,
                        toolbar_location=None)
    legend_glyph = legend_fig.circle(x=vals[0], y=vals[1], source=source,
                                     legend='Legend',
                                     color='Colours',
                                     fill_alpha=0.2,
                                     size=5
                                     )
    legend_fig.legend.location = 'top_left'
#    legend_fig.legend.location = (0, -30)
#    legend_fig.legend.visible = False
    legend_fig.legend.margin = 0
#    legend_fig.legend.orientation = 'horizontal'
    legend_fig.axis.visible = False
    legend_fig.grid.visible = False
    legend_fig.outline_line_color = None
    legend_glyph.visible = False
    fig_list.append(legend_fig)
#    legend_obj = legend_fig.legend[0]
#    legend_fig.legend.remove(legend_obj)
#    legend_fig.add_layout(legend_obj, 'right')

    # Get the number of grid columns from the rounded square root of number of
    # figures. But only use a maximum of 6 columns.
#    n_grid_cols = min(6, int(round(np.sqrt(len(fig_list)), 0)))
    n_grid_cols = min(6, int((np.sqrt(len(fig_list)))) + 1)
    # Create the final grid of figures
    grid = gridplot(fig_list, ncols=n_grid_cols, toolbar_location='left',
                    css_classes=['scrollable'],
#                    sizing_mode='scale_height',
#                    sizing_mode='scale_both',
#                    sizing_mode='stretch_both',
                    )

    return grid


def create_widgets_1(cats, cats_labels, colour_cat, filter_list, filter_true,
                     df, source):
    '''Create and return a list of the widgets for tab 1. There are two types
    of widgets:
        1. CheckboxButtonGroup: Used to toggle the category filters
        2. Select: A dropdown menu used to define the current colour category
        3. Button: Used to load a new file and then rebuild the whole layout

    Args:
        Basically everything...

    Return:
        wb_list_1 (List) : A list of widgetboxes, each containing widgets.

    '''
    cbg_list = []
    div_list = []
    for cat in cats:
        labels = cats_labels[cat]  # Lables of current category
        active_list = list(range(0, len(labels)))  # All labels start active
        cbg = CheckboxButtonGroup(labels=labels, active=active_list, width=999)
        cbg_list.append(cbg)

        # Make the annotation for the CheckboxButtonGroup:
        div = Div(text='''<div style="text-align:right; font-size:12pt;
                  border:6px solid transparent">'''+cat+''':</div>''',
                  width=250)
        div_list.append(div)

    for i, cbg in enumerate(cbg_list):
        # We need the update_filter function to know who calls it, so we use
        # the "partial" function to transport that information
        cbg.on_click(partial(update_filters, caller=i, cats=cats,
                             cats_labels=cats_labels, filter_list=filter_list,
                             filter_true=filter_true, df=df, source=source))

    sel = Select(title='Category label for colours:', value=colour_cat,
                 options=cats)
    sel.on_change('value', partial(update_colors, df=df,
                                   cats_labels=cats_labels, source=source))

    # Prepare the layout of the widgets:
    # Create rows with pairs of Div() and CheckboxButtonGroup(), where the
    # Div() contains the title. A list of those rows is combinded with a
    # widgetbox of the Select widget.
    row_list = zip(div_list, cbg_list)
    div_and_cgb_cols = []
    for row_new in row_list:
        div_and_cgb_cols.append(row(*row_new))

    wb_list_1 = [widgetbox(sel), div_and_cgb_cols]

    return wb_list_1


def create_widgets_2(filepath, vals, colour_cat, source, df, cats):
    '''Create and return a list of the widgets for tab 2. There are three
    types of widgets:
        1. Button: Used to load a new file and then rebuild the whole layout
        2. Div: Print text
        3. TextInput: Field for user input

    Args:
        filepath (str) : Path to file to load.

        ...

    Return:
        wb_list_2 (List) : A list of widgetboxes, each containing widgets.

    '''
    # First implementation
#    but_load_new = Button(label='Show file dialog', button_type='success')
#    but_load_new.on_click(load_file_dialog)

#    div1 = Div(text='''<div>
#                  Load a new file with a file dialog (only works if you run
#                  this script locally)
#                  </div>''', width=600)
#    div2 = Div(text='''<div> </div>''', height=33, width=600)  # Empty text
#
#    text_input = TextInput(value=filepath,
#                           title='Load this file (the server must have '
#                           'access to the file):',
#                           width=1000)
#
#    but_reload = Button(label='Load file', button_type='success')
#    but_reload.on_click(partial(reload_file, text_input))
#    wb = widgetbox(div1, but_load_new, div2, text_input, but_reload)

    # Second implementation
    div1 = Div(text='''<div style="position:relative; top:5px">
                  Upload a new file to the server:
                  </div>''', width=600)
    div2 = Div(text='''<div> </div>''', height=11, width=600)  # Empty text
    div3 = Div(text='''<div style="position:relative; top:15px">
               Select the value columns used in the plots:
               </div>''', height=25, width=600)
    div4 = Div(text='''<div> </div>''', height=11, width=600)  # Empty text

    save_path = os.path.join(os.path.dirname(__file__), 'upload')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    but_load_new = new_upload_button(save_path, load_file)

    sl_vals_max = Slider(start=2, end=len(vals), step=1,
                         value=min(vals_max, len(vals)),
                         title='Set the maximum number of value columns')

    sl_vals_max.on_change('value', update_vals_max)

    global ti_alert  # We need a global write access to this
    ti_alert = TextInput(value='',
                         title='Latest (error) message:',
                         width=1000)
    ti_alert.js_on_change('value', CustomJS(code='''alert(cb_obj.value)'''))

    active_list = list(range(0, min(len(vals), vals_max)))
    cg = CheckboxGroup(labels=vals, active=active_list)
    cg.on_change('active', partial(update_gridplot, vals=vals, source=source,
                                   df=df, colour_cat=colour_cat, cats=cats))

    wb = widgetbox(div1, but_load_new, div2, sl_vals_max,
                   div3, cg, div4, ti_alert)
    wb_list_2 = [wb]

    return wb_list_2


def create_data_table(source, df, cats, vals):
    '''Create and return the DataTable widget.
    TODO: Only show vals_active

    Args:
        source (ColumnDataSource) : The Bokeh source object

    Return:
        data_table (DataTable) : Bokeh DataTable widget.

    '''
    dt_columns = []

    for name in cats + vals:
        if df[name].dtype == 'datetime64[ns]':
            dt_column = TableColumn(field=name, title=name,
                                    formatter=DateFormatter())
        else:
            dt_column = TableColumn(field=name, title=name)
        dt_columns.append(dt_column)

    data_table = DataTable(source=source, columns=dt_columns,
                           fit_columns=True,
                           width=1400,
                           height=800,
                           scroll_to_selection=False,
                           sortable=True,  # editable=True,
                           )

    return data_table


def create_layout(wb_list_1, grid, wb_list_2, data_table, data_name):
    '''Create a Bokeh "layouts" from the widgetboxes and grid of figures.
    The layouts are organized into tabs and those are added as "root" to the
    current Bokeh document.

    Args:
        wb_list_1 (List) : A list of widgetboxes, each containing widgets.

        grid (Gridplot) : Grid containing all created figures.

        wb_list_2 (List) : A list of widgetboxes, each containing widgets.

        data_table (DataTable) : Bokeh DataTable widget.

        data_name (str) : The (file)name of the current data set.

    Return:
        None
    '''
    layout_1 = layout([wb_list_1, grid])
    layout_2 = layout(data_table)
    layout_3 = layout(wb_list_2)

    tab_1 = Panel(child=layout_1, title='Scatters')
    tab_2 = Panel(child=layout_2, title='Data Table')
    tab_3 = Panel(child=layout_3, title='Settings')
    tabs = Tabs(tabs=[tab_1, tab_2, tab_3])

    curdoc().clear()  # Clear any previous document roots
    curdoc().add_root(tabs)  # Add a new root to the document
    curdoc().title = 'DataExplorer: '+data_name

#    table_old = curdoc().roots[0].tabs[1].child
#    print(table_old)


def prepare_filter(cats, cats_labels, df):
    ''' Prepare the filters used to explore the data. A filter is a list of
    boolean values. Each category label needs its own filter. The filters start
    all 'True' and will be modified later, based on the user input.

    Args:
        cats (List) : List of the column names that contain categories.

        cats_labels (Dict) : Dictionary containing the categories associated
            with each label.

        df (Pandas DataFrame) : The input data we want to explore.

    Return:
        filter_list (List) : List containing all the filters, one for each
            category.
        filter_true (List) : A list with the length of one column, where all
            entries are "True".

    '''
    filter_list = []
    filter_true = []  # Define here, overwrite below, so we can use again later
    for cat in cats:
        labels = cats_labels[cat]
        filter_true = df[cat].isin(labels)
        filter_list.append(filter_true)

    return filter_list, filter_true


def update_filters(active, caller, cats, cats_labels,
                   filter_list, filter_true, df, source):
    '''Function associated with the CheckboxButtonGroups (CBG). Each CBG has
    one corresponding filter (which belongs to one category label). The calling
    CBG identifies itself with the "caller" argument. It delivers a list of the
    positions of the buttons which are now active (after the user input). The
    positions are translated into category strings (which are the button
    labels). The filters are updated based on the selected categories and then
    the DataFrame is filtered accordingly. Updating Bokeh's source object makes
    all the figures update, as well.

    Args:
        active (List) : A list of the positions of those buttons, which are
            currently active.

        caller (int) : Number of the ButtonGroup which is calling this function

        ... and lots more

    Returns:
        None

    '''
    i = caller
    cat_sel = cats[i]  # Name of category label corresponding to calling CBG
    labels = cats_labels[cat_sel]  # Categories within that label

    # Translate the active button positions into chosen category strings
    labels_active = []
    for j in active:
        labels_active.append(labels[j])

    # Get a boolean filter of the selected category label, where the selected
    # categories are "True". Store this in the correct filter_list
    filter_list[i] = df[cat_sel].isin(labels_active)

    # "Multiply" all filters to get one combined filter (Booleans are compared
    # with "&"). We start with all entries "True". Then we compare all filters
    # in the filter_list. In the end, only those rows remain "True" which are
    # "True" in all filters.
    filter_combined = filter_true
    for filter_i in filter_list:
        filter_combined = filter_combined & filter_i

    # Create a new ColumnDataSource object from the filtered DataFrame
    source_new = ColumnDataSource(data=df[filter_combined])

    # Update the "data" property of the "source" object with the new data.
    # Once the "source" changes, the figures and glyphs update automagically,
    # thanks to Bokeh's magic.
    source.data = source_new.data

    # The 'view' function seemed useful, but may not be flexible enough:
    # - Filtering one "column_name" for multiple "group"s seems not possible
    # - Changing the view did not seem to affect the DataTable
#    view.filters = [GroupFilter(column_name='Category Label 1', group='A'),
#                    GroupFilter(column_name='Category Label 2', group='First')
#                    ]
#    view.filters = [BooleanFilter(filter_combined)]


def update_colors(attr, old, new, df, cats_labels, source):
    '''Function associated with the colour category dropdown menu widget.
    The selected colour category label becomes the new "colour_cat".
    'Colours' is a column of both the DataFrame and the source which contains
    the current colours. It is updated, just like the 'Legend' column.

    Args:
        attr (str) : The widget's attribute that is told about in old and new

        old (str) : Previously selected colour category label.

        new (str) : Selected colour category label.

        df (Pandas DataFrame) : The input data we want to explore.

        cats_labels (Dict) : Dictionary containing the categories associated
            with each label.

        source (ColumnDataSource) : Bokeh's data format.

    Return:
        None

    '''
    colour_cat = new

    colourmap = get_colourmap(cats_labels[colour_cat])
    df['Legend'] = df[colour_cat]
    df['Colours'] = [colourmap[x] for x in df[colour_cat]]

    source.data['Legend'] = source.data[colour_cat]
    source.data['Colours'] = [colourmap[x] for x in source.data[colour_cat]]


def update_gridplot(attr, old, new, vals, source, df, colour_cat, cats):

    # Translate the active button positions into chosen category strings:
    vals_active = [vals[j] for j in new]

    if len(vals_active) > vals_max:
        message = 'Maximum of '+str(vals_max)+' value columns exceeded.'
        show_info(message)
        return
    elif len(vals_active) < 2:
        return
    else:
        # Create a new grid:
        grid_new = create_plots(source, df, vals_active, colour_cat)

        # Get the old grid and the layout containing it from current document:
        grid_old = curdoc().roots[0].tabs[0].child.children[1]
#        grid_old = curdoc().get_model_by_name('plot_grid')  # Does not work
        layout_1 = curdoc().roots[0].tabs[0].child

        # The children of a layout can be treated like a list:
        layout_1.children.remove(grid_old)
        layout_1.children.append(grid_new)

#        table_old = curdoc().roots[0].tabs[1].child.children[0]
#        print(table_old)
#        table_new = create_data_table(source, df, cats, vals_active)
#        layout_2 = curdoc().roots[0].tabs[1].child
#        layout_2.children.remove(table_old)
#        layout_2.children.append(layout(table_new))


def update_vals_max(attr, old, new):
    '''This function is triggered by the "sl_vals_max" slider widget.
    The user input value 'new' is stored in the global variable vals_max.

    Args:
        new (int) : User text input

    Return:
        None

    '''
    global vals_max
    vals_max = int(new)


def get_colourmap(categories):
    '''This function creates a dictionary of categories and their colours. It
    handles the possible exception thrown when the palette is not long enough.

    Args:
        categories (List) : List of categories.

    Return:
        colourmap (Dict) : Dictionary of categories and their colours.

    '''
    colourmap = dict()
    palette = palettes.all_palettes['Category20'][20]
#    palette = palettes.all_palettes['Spectral'][len(categories)]
#    palette = palettes.all_palettes['Spectral'][11]
    for i, cat in enumerate(categories):
        if i < 10:
            j = 2*i  # Even numbers
        else:
            j = 2*(i-9)-1  # Odd numbers
        try:
            colourmap[cat] = palette[j]
        except Exception:
            colourmap[cat] = 'grey'
    return colourmap


def load_file_dialog():
    '''This function is triggered by the "load new file" button and presents
    a file dialog. The user choice is handed to the load_file() function.

    Args:
        None

    Return:
        None

    '''
    cwd = os.getcwd()
#    root = Tk()
#    dirname = filedialog.askdirectory(parent=root,
#                                        initialdir=cwd,
#                                        title='Please select a directory')
#    if len(dirname) > 0:
#        print("You chose %s" % dirname)

    root = Tk()
    root.withdraw()  # Important: Hides the empty 'root' window
    root.filename = filedialog.askopenfilename(initialdir=cwd,
                                               title="Please choose your file",
                                               filetypes=(
                                                       ("Excel", "*.xlsx"),
                                                       ("all files", "*.*"),
                                                       )
                                               )
    load_file(root.filename)


def reload_file(ti_alert):
    '''This function is triggered by the "load file" button. It asks the
    ti_alert widget for its current value and hands that to load_file().

    Args:
        ti_alert (TextInput) : Bokeh widget.

    Return:
        None

    '''
    filepath = ti_alert.value
    load_file(filepath)


def load_file(filepath):
    '''The chosen file is read into a Pandas DataFrame.
    Supported file types are '.xlsx' and '.xls'. Pandas will also try to read
    in '.csv' files, but can easily fail if the separators are not guessed
    correctly.
    In order to regenerate all widgets and figures, create_dataexplorer_UI()
    is called. This finishes with calling create_layout(), which "cleares"
    the current Bokeh dokument and adds a new root to the empty document.

    Args:
        filepath (str) : Path to file to load.

    Return:
        None

    '''
    if len(filepath) == 0:  # No file selected, or file dialog cancelled
        return  # Return, instead of completing the function

    logging.info('Trying to open file: ' + filepath)
    try:
        filetype = os.path.splitext(os.path.basename(filepath))[1]
        if filetype in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif filetype in ['.csv']:
            df = pd.read_csv(filepath,
                             sep=None, engine='python',  # Guess separator
                             parse_dates=[0],  # Try to parse first col as date
                             infer_datetime_format=True)
        else:
            raise NotImplementedError('Unsupported file extension: '+filetype)
    except Exception as ex:
        # Show the error message in the terminal and in a pop-up messagebox:
        show_info('File not loaded: '+filepath+' \n'+str(ex))
        return  # Return, instead of completing the function

    logging.debug('Loaded ' + filepath)

    data_name = os.path.basename(filepath)
    create_dataexplorer_UI(df, filepath, data_name)


def show_info(message):
    '''Shows a notification window with given title and message.

    Args:
        message (str) : Message text.

    Return:
        None

    '''
    global ti_alert
    try:  # The ti_alert widget may not have been created yet
        timestamp = pd.datetime.now().time().strftime('%H:%M')
        ti_alert.value = timestamp + ' ' + message
    except Exception:
        pass
    logging.critical(message)  # Bokeh's server logging funtion


'''
Main function:

The following lines are executed when the python script is started by the
bokeh server. We create an initial set of test data and then create the
DataExplorer user interface.
'''

vals_max = 6  # Default for the global variable of maximum value columns
df = create_test_data()
data_name = 'Test Data'
filepath = r'\\igs-srv\transfer\Joris_Nettelstroth\Python\DataExplorer' + \
           '\excel_text.xlsx'

create_dataexplorer_UI(df, filepath, data_name)
