# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 08:10:21 2017

@author: nettelstroth

Start this script with:
bokeh serve --show dataexplorer.py

"""

import pandas as pd
import numpy as np
#from bokeh.io import output_file, show
from bokeh.layouts import widgetbox, gridplot, layout#, row
from bokeh.models.widgets import CheckboxButtonGroup, Select, Button
from bokeh.models.widgets import Div
from bokeh.models import ColumnDataSource#, CDSView, GroupFilter
from bokeh.models import CategoricalColorMapper, Legend
from bokeh.models import Circle
from bokeh.server.connection import ServerConnection
from bokeh.plotting import figure
from bokeh.palettes import Spectral4 as palette
from bokeh.io import curdoc
from functools import partial
import itertools
from tkinter import Tk, filedialog
import os


"""
Create some test data
"""
def create_test_data():
    time_steps = 10
    
    new_index = pd.date_range(start='2017-11-01 00:00',
    #                          end='2017-11-05 00:00',
                              periods=time_steps,
                              freq='D')
    
    dataA1 = {'T1': np.random.randint(0, 20, time_steps),
              'T2': np.random.randint(0, 30, time_steps),
              'Sin': np.sin(np.linspace(-np.pi, np.pi, time_steps)),
              'Category Group 1': pd.Categorical(['A']*time_steps),
              'Category Group 2': pd.Categorical(['First']*time_steps),
              'Category Group 3': pd.Categorical(['10-20']*time_steps),
              }
    dataA2 = {'T1': np.random.randint(10, 30, time_steps),
              'T2': np.random.randint(10, 40, time_steps),
              'Sin': np.sin(np.linspace(-2*np.pi, 2*np.pi, time_steps)),
              'Category Group 1': pd.Categorical(['A']*time_steps),
              'Category Group 2': pd.Categorical(['Second']*time_steps),
              'Category Group 3': pd.Categorical(['10-20']*time_steps),
              }
    dataB1 = {'T1': np.random.randint(20, 40, time_steps),
              'T2': np.random.randint(20, 50, time_steps),
              'Sin': np.sin(np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Group 1': pd.Categorical(['B']*time_steps),
              'Category Group 2': pd.Categorical(['First']*time_steps),
              'Category Group 3': pd.Categorical(['10-20']*time_steps),
              }
    dataB2 = {'T1': np.random.randint(30, 50, time_steps),
              'T2': np.random.randint(30, 60, time_steps),
              'Cos': np.cos(np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Group 1': pd.Categorical(['B']*time_steps),
              'Category Group 2': pd.Categorical(['Third']*time_steps),
              'Category Group 3': pd.Categorical(['20-30']*time_steps),
              }
    dataC1 = {'T1': np.random.randint(40, 60, time_steps),
              'T2': np.random.randint(40, 70, time_steps),
              'Sin': np.sin(0.5*np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Group 1': pd.Categorical(['C']*time_steps),
              'Category Group 2': pd.Categorical(['Second']*time_steps),
              'Category Group 3': pd.Categorical(['20-30']*time_steps),
              }
    dataC2 = {'T1': np.random.randint(50, 70, time_steps),
              'T2': np.random.randint(50, 80, time_steps),
              'Cos': np.cos(0.5*np.linspace(-3*np.pi, 3*np.pi, time_steps)),
              'Category Group 1': pd.Categorical(['C']*time_steps, categories=['C']),
              'Category Group 2': pd.Categorical(['Third']*time_steps),
              'Category Group 3': pd.Categorical(['20-30']*time_steps),
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
    df.reset_index(level=0, inplace=True)
    #df.to_excel('excel_text.xlsx')
    #print(df)
    
    return df

"""
Extract the column names from the data
"""

def analyse_dataframe(df):
    '''Analyse a given DataFrame to seperate the columns in categories and
    values. "Object" columns become categories, their column names are saved
    as category labels.
    
    Args:
        df (Pandas DataFrame):
            The input data we want to explore.
    
    Returns:
        cats (List):
            List of categories found in the DataFrame.
        cats_labels (Dict):
            Dictionary 
    
    '''
    columns_all = df.columns.values.tolist()
    cats = []
    vals = []
    for column in columns_all:
        if df[column].dtype == 'object':
            cats.append(column)
            df[column] = df[column].astype('category')
        else:
            vals.append(column)
    
    cats_labels = dict()
    for cat in cats:
    #    print(cat, (list(set(df[cat]))))
        cats_labels[cat] = sorted(list(set(df[cat])))
        
    return cats, cats_labels, vals




#print(df["Cat1"])
#df["Cat1"] = df["Cat1"].astype('category')
#print(df['Cat1'])
#print(df)


"""
Use Bokeh to plot the data in an interactive way
"""

#df = create_test_data()
#cats, vals, cats_labels = analyse_dataframe(df)
#colour_cat = cats[0]

def create_plots(df, vals, colour_cat):
    source2 = ColumnDataSource(data=df)
    
    # The 'view' function seemed useful at first, but may not be flexible enough
    #source_filters = [
    #                  GroupFilter(column_name='Cat1', group='A'),
    #                  GroupFilter(column_name='Cat2', group='First'),
    #                  ]
    #
    #view = CDSView(source=source2,
    #               filters=source_filters)
    
    
    plot_size_and_tools = {'tools': ['pan', 'wheel_zoom', 'box_zoom', 'box_select',
                                     'lasso_select', 'reset', 'help'],
                           'active_scroll': 'wheel_zoom',
    #                       'toolbar_location': 'left',
                           'plot_height': 250, 'plot_width': 250,
                           }
    
    
    bokeh_colormap = CategoricalColorMapper(palette=palette,
                                            factors=list(set(df[colour_cat])))
    colour_def = {'field': colour_cat, 'transform': bokeh_colormap}
    
    
    # A choice of combinatoric generators with decending number of results:
    #permutation_list = itertools.product(vals, repeat=2)
    #permutation_list = itertools.permutations(vals, r=2)
    #permutation_list = itertools.combinations_with_replacement(vals, r=2)
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
    
        glyph = p.circle(x=x_val, y=y_val, source=source2,
                         legend=colour_cat, color=colour_def,
                         )
        p.xaxis.axis_label = x_val
        p.yaxis.axis_label = y_val
        p.legend.visible = False
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        fig_list.append(p)
        glyph_list.append(glyph)
    #    p.legend.items = Legend(items=[('test', [glyph])])
    #    p.legend.items = LegendItem=[('test', [glyph])]
    
    # Get the number of grid columns from the rounded square root of number of figs
    n_grid_cols = int(round(np.sqrt(len(fig_list)), 0))
    grid = gridplot(fig_list, ncols=n_grid_cols, toolbar_location='right')
    
    return grid, glyph_list, source2


"""
Test for adding legend
"""
#legend_fig = figure(plot_height= 150, plot_width= 150,
#                    toolbar_location=None)
#fake_glyph = legend_fig.circle(x=vals[0], y=vals[1], source=source2,
#                 legend=colour_cat,
#                 color=colour_def,
#                 name='fake_glyph',
#                 )
#legend_fig.axis.visible = False
#
#legend = Legend(items=[('X', [glyph]),
#                       ],
#                location=(0,-30)
#                )
#


"""
Widget function definitions
"""

def prepare_filter(cats, cats_labels):
    # Prepare a list containing all the filters, one for each category
    # The filters start all 'True' and will be modified later
    filter_list = []
    filter_true = []  # Define here, overwrite below, so we can use again later
    for cat in cats:
        labels = cats_labels[cat]
        filter_true = df[cat].isin(labels)
        filter_list.append(filter_true)
        
    return filter_list, filter_true


def update_filter(filter_list, filter_true, source2):
    filter_combined = filter_true
    # "Multiply" all filters, to get one combined filter
    # (Booleans are compared with "&")
    for filter_i in filter_list:
        filter_combined = filter_combined & filter_i
#    print(filter_combined)
    source_new = ColumnDataSource(data=df[filter_combined])
    source2.data = source_new.data


def update_filter_i(active, caller, cats, cats_labels,
                    filter_list, filter_true, source2):
#    global filter_list
    i = caller
    cat_sel = cats[i]
    labels = cats_labels[cat_sel]

    labels_active = []
    for j in active:
        labels_active.append(labels[j])

    filter_list[i] = df[cat_sel].isin(labels_active)
    update_filter(filter_list, filter_true, source2)


def update_colors(attr, old, new, glyph_list):
    colour_cat = new
    bokeh_colormap = CategoricalColorMapper(palette=palette,
                                            factors=list(set(df[colour_cat])))
    colour_def = {'field': colour_cat, 'transform': bokeh_colormap}
    for gly in glyph_list:
        gly.glyph.fill_color = colour_def
        gly.glyph.line_color = colour_def
        
        
    """
    Test for modifying legend
    """
#    for fig in fig_list:
#        gly = fig.select_one({'name': 'glyph_name'})
#        gly.glyph.fill_color = colour_def
#        gly.glyph.line_color = colour_def
        
#    for fig in fig_list:
#        fig.legend = Legend(colour_cat)
        
#    global legend_fig
#    global fake_glyph
#    legend_fig.axis.visible = True
#    fake_glyph = legend_fig.select_one({'name': 'fake_glyph'})
#    legend_fig.renderers.remove(fake_glyph)
#    legend_fig.circle(x=vals[0], y=vals[1], source=source2,
#                     legend=colour_cat,
#                     color=colour_def,
#                     name='fake_glyph',
#                     )
#    legend_fig.add_glyph(source2,
#                         Circle(x=vals[0], y=vals[1],
##                                legend=colour_cat,
#                                line_color=colour_def,
#                                fill_color=colour_def,
#                                name='fake_glyph',
#                                )
#                         )


def load_file():
    global df
    cwd = os.getcwd()
    print(cwd)
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
                                                ("all files", "*.*"),
                                                ("Excel", "*.xlsx"),
                                                )
                                               )
    print(root.filename)
#    print(os.path.exists(root.filename))
    df = pd.read_excel(root.filename)
#    source_new = ColumnDataSource(data=df)
#    source2.data = source_new.data
    
#    print(curdoc().roots)
    
    curdoc().clear()
    
    create_dataexplorer_UI(df)
    
#    cats, vals, cats_labels = analyse_dataframe(df)
#    colour_cat = cats[0]
#    
#    grid, fig_list, glyph_list, source2 = create_plots(df, cats, vals, colour_cat)
#    
#    filter_list, filter_true = prepare_filter(cats, cats_labels)
#    
#    wb_list = update_widgets(cats, cats_labels, colour_cat)
#    update_layout(wb_list, grid)
    
#    ServerConnection.send_patch_document('Hello')
#    curdoc().detach_session()




"""
Widget definitions
"""
def update_widgets(cats, cats_labels, colour_cat, filter_list, filter_true,
                   source2, glyph_list):
    cbg_list = []
    div_list = []
    for cat in cats:
        labels = cats_labels[cat]  # Lables of current category
        active_list = list(range(0, len(labels)))  # All labels start active
        cbg = CheckboxButtonGroup(labels=labels, active=active_list)
        cbg_list.append(cbg)
        
        # Make the annotation for the CheckboxButtonGroup:
        div = Div(text="""<div style="text-align:right;font-size:12pt">
                  """+cat+""":
                  </div>""", height=33)
        div_list.append(div)
    
    for i, cbg in enumerate(cbg_list):
        # We need the update_filter function to know who calls it, so we use
        # the "partial" function to transport that information
        cbg.on_click(partial(update_filter_i, caller=i, cats=cats,
                             cats_labels=cats_labels,
                             filter_list=filter_list, filter_true=filter_true,
                             source2=source2
                             ))
    
    categories = list(cats_labels.keys())
#    categories = cats
    sel = Select(title='Category group for colours:', value=colour_cat,
                 options=categories)
    sel.on_change('value', partial(update_colors, glyph_list=glyph_list))
    
    but_load = Button(label='Load new file', button_type='success')
    but_load.on_click(load_file)


    """
    Layout of widgets and plots
    """
    
    #wb_list = []
    #for i, cbg in enumerate(cbg_list):
    #    wb = widgetbox(div_list[i], cbg)
    #    wb_list.append(wb)
    
    wb1 = widgetbox(*div_list, width=200)
    wb2 = widgetbox(*cbg_list)
    wb3 = widgetbox(sel, but_load)
    
    wb_list = [wb1, wb2, wb3]
    
    return wb_list


def update_layout(wb_list, grid):
    layout1 = layout(wb_list, [grid])

    # For standalone html output:
    #output_file('dataexplorer.html', title='Data Explorer')
    #show(l)
    
    # For Bokeh Server use:
    # add layout to the current document
    curdoc().add_root(layout1)
    curdoc().title = 'Data Explorer'





def create_dataexplorer_UI(df):
    '''Perform all the tasks necessary to create the Data Explorer user
    interface. Is also used to re-create the UI when new data is loaded.
    
    Args:
        df (Pandas DataFrame):
            The input data we want to explore.
            
    Returns:
        None
    
    '''
    
    # Get categories, their labels and vals (column names of values) from df
    cats, cats_labels, vals = analyse_dataframe(df)
    colour_cat = cats[0]
        
    grid, glyph_list, source2 = create_plots(df, vals, colour_cat)
    
    filter_list, filter_true = prepare_filter(cats, cats_labels)
    wb_list = update_widgets(cats, cats_labels, colour_cat, filter_list,
                             filter_true, source2, glyph_list)
    
    update_layout(wb_list, grid)


"""
Main function:
    
The following lines are executed when the python script is started by the
bokeh server. We create an initial set of test data and then create the 
Data Explorer UI.
"""
df = create_test_data()
create_dataexplorer_UI(df)







