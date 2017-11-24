# -*- coding: utf-8 -*-
'''
Created on Fri Nov 24 09:54:53 2017

@author: nettelstroth

Additional helper functions for the script dataexplorer.py

'''

import os
import logging
import base64
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.models.widgets import Button


def new_upload_button(save_path, callback, label="Upload file"):
    '''
    A button widget that implements a special javascript callback function.
    This callback function makes the browser open a file dialog and allows
    the user to select a file, which is then uploaded to the folder 'save_path'
    on the server.
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
        logging.info("New file uploaded: " + file_path)
        callback(file_path)

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
