# -*- coding: utf-8 -*-
'''
Created on Fri Nov  3 08:10:21 2017

@author: Joris Nettelstroth

The DataExplorer should help you explore correlations within your data. It
features a user interface that shows scatter plots of all the variables in
your data. Categories found in your data can be used to filter the views.

Start this program with a command prompt in Windows:
    - Go to Start and type "cmd"
    - Select "Eingabeaufforderung"
    - Change directory to the folder containing the script by typing
      "cd path/to/folder"
    - Start a Bokeh server running this script by typing
      "bokeh serve dataexplorer.py --show"
    - Your webbrowser should open and display the interface of the program
    - Hit the button to load your own Excel file

If you do not yet have Python and Bokeh installed, the easiest way to do that
is by downloading and installing "Anaconda" from here:
https://www.anaconda.com/download/
Its a package manager that distributes Python with data science packages.

'''

import pandas as pd
import numpy as np
import itertools
import os
from bokeh.layouts import widgetbox, gridplot, layout  # , row
from bokeh.models.widgets import CheckboxButtonGroup, Select, Button
from bokeh.models.widgets import Div
from bokeh.models import ColumnDataSource, CategoricalColorMapper
# from bokeh.models import Legend, CDSView, GroupFilter
# from bokeh.models import Circle
# from bokeh.server.connection import ServerConnection
from bokeh.plotting import figure
from bokeh.palettes import Spectral4 as palette
from bokeh.io import curdoc
from functools import partial
from tkinter import Tk, filedialog


def create_test_data():
    '''Create some test data. Includes sin, cos and some random numbers. The
    amount of data is hardcoded and controlled with the number of time steps.

    Args:
        None

    Returns:
        df (Pandas DataFrame) : An example set of test data.
    '''

    time_steps = 10  # Control the amount of test data

    new_index = pd.date_range(start=pd.to_datetime('today'),
                              periods=time_steps, freq='D')

    dataA1 = {'T1': np.random.randint(0, 20, time_steps),
              'T2': np.random.randint(0, 30, time_steps),
              'Sin': np.sin(np.linspace(-np.pi, np.pi, time_steps)),
              'Category Label 1': pd.Categorical(['A']*time_steps),
              'Category Label 2': pd.Categorical(['First']*time_steps),
              'Category Label 3': pd.Categorical(['10-20']*time_steps),
              }
    dataA2 = {'T1': np.random.randint(10, 30, time_steps),
              'T2': np.random.randint(10, 40, time_steps),
              'Sin': np.sin(np.linspace(-2*np.pi, 2*np.pi, time_steps)),
              'Category Label 1': pd.Categorical(['A']*time_steps),
              'Category Label 2': pd.Categorical(['Second']*time_steps),
              'Category Label 3': pd.Categorical(['10-20']*time_steps),
              }
    dataB1 = {'T1': np.random.randint(20, 40, time_steps),
              'T2': np.random.randint(20, 50, time_steps),
              'Sin': np.sin(np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Label 1': pd.Categorical(['B']*time_steps),
              'Category Label 2': pd.Categorical(['First']*time_steps),
              'Category Label 3': pd.Categorical(['10-20']*time_steps),
              }
    dataB2 = {'T1': np.random.randint(30, 50, time_steps),
              'T2': np.random.randint(30, 60, time_steps),
              'Cos': np.cos(np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Label 1': pd.Categorical(['B']*time_steps),
              'Category Label 2': pd.Categorical(['Third']*time_steps),
              'Category Label 3': pd.Categorical(['20-30']*time_steps),
              }
    dataC1 = {'T1': np.random.randint(40, 60, time_steps),
              'T2': np.random.randint(40, 70, time_steps),
              'Sin': np.sin(0.5*np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Label 1': pd.Categorical(['C']*time_steps),
              'Category Label 2': pd.Categorical(['Second']*time_steps),
              'Category Label 3': pd.Categorical(['20-30']*time_steps),
              }
    dataC2 = {'T1': np.random.randint(50, 70, time_steps),
              'T2': np.random.randint(50, 80, time_steps),
              'Cos': np.cos(0.5*np.linspace(-3*np.pi, 3*np.pi, time_steps)),
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

#    df.to_excel('excel_text.xlsx')  # Save this as an Excel file if you want
#    print(df)  # Show the final DataFrame in the terminal window

    return df


def create_dataexplorer_UI(df, data_name):
    '''Performs all the tasks necessary to create the Data Explorer user
    interface. It is also used to re-create the UI when new data is loaded.

    Args:
        df (Pandas DataFrame) : The input data we want to explore.

        data_name (str) : The (file)name of the current data set.

    Returns:
        None

    '''

    # Get categories, their labels and vals (column names of values) from df
    cats, cats_labels, vals = analyse_dataframe(df)
    colour_cat = cats[0]  # The first colour category label is the default

    # Use Bokeh to plot the data in an interactive way
    grid, glyph_list, source = create_plots(df, cats_labels, vals, colour_cat)

    # Prepare the filters used to explore the data
    filter_list, filter_true = prepare_filter(cats, cats_labels, df)

    # Create and get a list of all the widgets used for user interaction
    wb_list = create_widgets(cats, cats_labels, colour_cat, filter_list,
                             filter_true, df, source, glyph_list)

    # Create a Bokeh "layout" from the widgetboxes and grid of figures
    create_layout(wb_list, grid, data_name)

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
    '''

    columns = df.columns.values.tolist()
    cats = []
    vals = []
    for column in columns:
        # The column contains categories or values
        if df[column].dtype == 'object':
            cats.append(column)
#            df[column] = df[column].astype('category')
        else:
            vals.append(column)

    cats_labels = dict()
    for cat in cats:
        cats_labels[cat] = sorted(list(set(df[cat])))
#        print(cat, (list(set(df[cat]))))

    return cats, cats_labels, vals


def create_plots(df, cats_labels, vals, colour_cat):
    '''Use Bokeh to plot the data in an interactive way. The Bokeh settings
    are defined in this function, as well as the combinatoric generator which
    generates the combinations of all the "vals". For each combination, a
    figure is created and all of those figures are combined into one grid.

    Args:
        df (Pandas DataFrame) : The input data we want to explore.

        cats_labels (Dict) : Dictionary containing the categories associated
            with each label.

        vals (List) : List of the column names that contain values.

        colour_cat (str) : Name of the current colour category label.

    Returns:
        grid (Gridplot) : Grid containing all created figures.

        glyph_list (List) : List of all used glyphs.

        source (ColumnDataSource) : Bokeh's data format.

    '''
    source = ColumnDataSource(data=df)

    # The 'view' function seemed useful, but may not be flexible enough
#    source_filters = [
#                      GroupFilter(column_name='Cat1', group='A'),
#                      GroupFilter(column_name='Cat2', group='First'),
#                      ]
#
#    view = CDSView(source=source,
#                   filters=source_filters)

    plot_size_and_tools = {'tools': ['pan', 'wheel_zoom', 'box_zoom', 'reset',
                                     'lasso_select', 'box_select', 'help'],
                           'active_scroll': 'wheel_zoom',
                           'plot_height': 250, 'plot_width': 250,
                           }

    bokeh_colormap = CategoricalColorMapper(palette=palette,
                                            factors=cats_labels[colour_cat])
    colour_def = {'field': colour_cat, 'transform': bokeh_colormap}

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
        if x_val == 'Time':
            p = figure(**plot_size_and_tools, x_axis_type='datetime')
        elif y_val == 'Time':
            p = figure(**plot_size_and_tools, y_axis_type='datetime')
        else:
            p = figure(**plot_size_and_tools)

        glyph = p.circle(x=x_val, y=y_val, source=source,
                         legend=colour_cat, color=colour_def,
                         )
        p.xaxis.axis_label = x_val
        p.yaxis.axis_label = y_val
        p.legend.visible = False
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        fig_list.append(p)
        glyph_list.append(glyph)
#        p.legend.items = Legend(items=[('test', [glyph])])
#        p.legend.items = LegendItem=[('test', [glyph])]

    # Get the number of grid columns from the rounded square root of number of
    # figures
    n_grid_cols = int(round(np.sqrt(len(fig_list)), 0))
    # Create the final grid of figures
    grid = gridplot(fig_list, ncols=n_grid_cols, toolbar_location='right')

    return grid, glyph_list, source

    '''
    Test for adding legend
    '''
#    legend_fig = figure(plot_height= 150, plot_width= 150,
#                        toolbar_location=None)
#    fake_glyph = legend_fig.circle(x=vals[0], y=vals[1], source=source,
#                     legend=colour_cat,
#                     color=colour_def,
#                     name='fake_glyph',
#                     )
#    legend_fig.axis.visible = False
#
#    legend = Legend(items=[('X', [glyph]),
#                           ],
#                    location=(0,-30)
#                    )
#


def create_widgets(cats, cats_labels, colour_cat, filter_list, filter_true,
                   df, source, glyph_list):
    '''Create and return a list of all the widgets used for user interaction.
    There are three types of widgets:
        1. CheckboxButtonGroup: Used to toggle the category filters
        2. Select: A dropdown menu used to define the current colour category
        3. Button: Used to load a new file and then rebuild the whole layout

    Args:
        Basically everything...

    Return:
        wb_list (List) : A list of widgetboxes, each containing widgets.

    '''
    cbg_list = []
    div_list = []
    for cat in cats:
        labels = cats_labels[cat]  # Lables of current category
        active_list = list(range(0, len(labels)))  # All labels start active
        cbg = CheckboxButtonGroup(labels=labels, active=active_list)
        cbg_list.append(cbg)

        # Make the annotation for the CheckboxButtonGroup:
        div = Div(text='''<div style="text-align:right;font-size:12pt">
                  '''+cat+''':
                  </div>''', height=33)
        div_list.append(div)

    for i, cbg in enumerate(cbg_list):
        # We need the update_filter function to know who calls it, so we use
        # the "partial" function to transport that information
        cbg.on_click(partial(update_filters, caller=i, cats=cats,
                             cats_labels=cats_labels,
                             filter_list=filter_list, filter_true=filter_true,
                             df=df, source=source
                             ))

    sel = Select(title='Category label for colours:', value=colour_cat,
                 options=cats)
    sel.on_change('value', partial(update_colors, df=df,
                                   glyph_list=glyph_list))

    but_load = Button(label='Load new file', button_type='success')
    but_load.on_click(load_file)

    # Put all the widgets in boxes, so they can be handled more easily:
    wb1 = widgetbox(*div_list, width=200)
    wb2 = widgetbox(*cbg_list)
    wb3 = widgetbox(sel, but_load)
    wb_list = [wb1, wb2, wb3]

    return wb_list


def create_layout(wb_list, grid, data_name):
    '''Create a Bokeh "layout" from the widgetboxes and grid of figures and
    add it as "root" to the current Bokeh document.

    Args:
        wb_list (List) : A list of widgetboxes, each containing widgets.

        grid (Gridplot) : Grid containing all created figures.

        data_name (str) : The (file)name of the current data set.

    Return:
        None
    '''
    layout1 = layout(wb_list, [grid])
    curdoc().add_root(layout1)
    curdoc().title = 'DataExplorer: '+data_name


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


def update_colors(attr, old, new, df, glyph_list):
    '''Function associated with the colour category dropdown menu widget.
    The selected colour category label becomes the new "colour_cat". Based on
    that the Bokeh colourmap is regenerated and applied to all the glyphs.

    Args:
        attr (str) : The widget's attribute that is told about in old and new

        old (str) : Previously selected colour category label.

        new (str) : Selected colour category label.

        df (Pandas DataFrame) : The input data we want to explore.

        glyph_list (List) : List of all used glyphs.

    Return:
        None

    '''
    colour_cat = new
    bokeh_colormap = CategoricalColorMapper(palette=palette,
                                            factors=list(set(df[colour_cat])))
    colour_def = {'field': colour_cat, 'transform': bokeh_colormap}
    for gly in glyph_list:
        gly.glyph.fill_color = colour_def
        gly.glyph.line_color = colour_def

    '''
    Test for modifying legend
    '''
#    for fig in fig_list:
#        gly = fig.select_one({'name': 'glyph_name'})
#        gly.glyph.fill_color = colour_def
#        gly.glyph.line_color = colour_def
#
#    for fig in fig_list:
#        fig.legend = Legend(colour_cat)
#
#    global legend_fig
#    global fake_glyph
#    legend_fig.axis.visible = True
#    fake_glyph = legend_fig.select_one({'name': 'fake_glyph'})
#    legend_fig.renderers.remove(fake_glyph)
#    legend_fig.circle(x=vals[0], y=vals[1], source=source,
#                     legend=colour_cat,
#                     color=colour_def,
#                     name='fake_glyph',
#                     )
#    legend_fig.add_glyph(source,
#                         Circle(x=vals[0], y=vals[1],
#                                line_color=colour_def,
#                                fill_color=colour_def,
#                                name='fake_glyph',
#                                legend=colour_cat,
#                                )
#                         )


def load_file():
    '''This function is triggered by the "load new file" button and presents
    a file dialog. The chosen file is read into a Pandas DataFrame. In order
    to regenerate all widgets and figures, the current Bokeh dokument has to
    be "cleared". Then create_dataexplorer_UI() is called which will add a
    new root to the empty document.

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
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir=cwd,
                                               title="Please choose your file",
                                               filetypes=(
                                                       ("Excel", "*.xlsx"),
                                                       ("all files", "*.*"),
                                                       )
                                               )

    try:
        df = pd.read_excel(root.filename)
    except Exception as ex:
        # TODO this message has to be visible in the HTML document
        print('File '+root.filename+' NOT loaded!')
        print(ex)
        return  # Return before completing the function

#    print(curdoc().roots)
    curdoc().clear()
    data_name = os.path.basename(root.filename)
    create_dataexplorer_UI(df, data_name)


'''
Main function:

The following lines are executed when the python script is started by the
bokeh server. We create an initial set of test data and then create the
DataExplorer user interface.
'''
df = create_test_data()
data_name = 'Test Data'
create_dataexplorer_UI(df, data_name)
