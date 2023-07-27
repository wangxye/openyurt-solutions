#!/usr/bin/env python3

import os
import sys
import logging
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/pipeline', methods=['POST'])
def pipeline():
    url = ""
    dev = "CPU"
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()
        if "url" in json:
            url = json["url"]
        else:
            return "url must not be null"
        if "dev" in json:
            dev = json["dev"]
        pipeline_cmd = f'gst-launch-1.0 -v filesrc location={url} ! decodebin ! videoconvert n-threads=4 ! \
capsfilter caps="video/x-raw,format=BGRx" ! gvainference model-instance-id=nunu model="model" device={dev} ! \
queue ! gvafpscounter ! fakesink async=false'
        os.system(pipeline_cmd)
        return pipeline_cmd
    else:
        return 'Content-Type not supported!'

if __name__ == '__main__':
    app.run()
