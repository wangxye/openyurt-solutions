#!/usr/bin/env python3

import os
from flask import Flask, request

'''
    pipeline server run with dlstreamer
    port: 55555
'''

workDir = "/home/dlstreamer/"
videoDir = workDir + "videos/"
modelDir = workDir + "models/"

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/pipeline', methods=['POST'])
def pipeline():
    src = "rtspsrc"
    url = ""
    dev = "CPU"
    model = "horizontal-text-detection-0001.xml"

    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()
        if "url" in json:
            url = json["url"]
        else:
            return "url must not be null"
        if "src" in json:
            src = json["src"]
        if "dev" in json:
            dev = json["dev"]
        if "model" in json:
            model = json["model"]

        pipeline_cmd = create_pipeline(src=src, url=url, model=model, dev=dev)
        return pipeline_cmd + " created"
    else:
        return 'Content-Type not supported!'

def create_pipeline(src, url, model, dev):
    model = modelDir + model
    if src == "filesrc":
        url = videoDir + url

    pipeline_cmd = f'gst-launch-1.0 -v {src} location={url} ! decodebin ! videoconvert n-threads=4 ! \
capsfilter caps="video/x-raw,format=BGRx" ! gvainference model-instance-id=nunu model={model} device={dev} ! \
queue ! gvafpscounter ! fakesink async=false'
    pid = os.fork()
    if pid == 0:
        os.system(pipeline_cmd)
    return pipeline_cmd

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=55555)
