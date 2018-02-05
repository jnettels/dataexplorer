# DataExplorer #

The DataExplorer should help you explore correlations within your data. It
features a user interface that shows scatter plots of all the variables in
your data. Classes found in your data can be used to filter the views.

### How do I get set up? ###

You can start this program with the `start_bokeh.cmd` batch file. This starts
a Bokeh server that can be accessed from any computer in the network.
 
Alternatively, you can start it with your own command prompt in Windows:

* Go to Start and type `cmd`
* Select `Eingabeaufforderung`
* Change to the directory containing the folder `dataexplorer` (the folder
  including the files `main.py` and `dataexplorer.py`) with the command:
  `cd path/to/folder`
* Start a Bokeh server running this application by typing:
  `bokeh serve dataexplorer --show`
* Your webbrowser should open and display the interface of the program
* Under settings, hit the button to load your own Excel file

The file `excel_example.xlsx` shows the required input format and gives hints
about the usage.
 
If you do not yet have Python and Bokeh installed, the easiest way to do that
is by downloading and installing `Anaconda` from 
[here](https://www.anaconda.com/download/).
It's a package manager that distributes Python with data science packages.
 
During installation, please allow to add variables to `$PATH` (or do that
manually afterwards). This allows Bokeh to be started from everywhere, which
is required for the batch file to work.

### Who do I talk to? ###

If in trouble, contact Joris Nettelstroth