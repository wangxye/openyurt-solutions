#!/usr/bin/env python3

import os
from flask import Flask, request,Response
from realtime_objectdetection import *
import asyncio

'''
    pipeline server run with dlstreamer
    port: 55555
'''

workDir = "/home/dlstreamer/"
videoDir = workDir + "openyurt-solutions/onvif/tests/"
modelDir = workDir + "models/"
video_url = videoDir + "Road-vehicle-run.mp4"

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/inference_video', methods=['GET'])
def inference_video():
    return Response(gen_inference_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_inference_frame():
    global last_frame
    while True:
        # get the final image
        # frame = last_frame
        frame = frame_queue.get()
        # encode the image to JPEG format
        _, encoded_img = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + 
               bytearray(encoded_img) + b"\r\n")
  
        
@app.route('/pipeline', methods=['POST'])
def pipeline():
    src = "rtspsrc"
    url = ""
    dev = "CPU"
    model = "person-vehicle-bike-detection-2000.xml"

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
    global video_url
    if src == "filesrc":
        url = videoDir + url
        video_url = url
    else:
        video_url = url
    print(video_url)
    
    pipeline_cmd = f'gst-launch-1.0 -v {src} location={url} ! decodebin ! videoconvert n-threads=4 ! \
capsfilter caps="video/x-raw,format=BGRx" ! gvainference model-instance-id=nunu model={model} device={dev} ! \
queue ! gvafpscounter ! fakesink async=false'   

    def run_detection():
        run_object_detection(video_url, False, True, 0, True)

    detection_thread = threading.Thread(
        target=run_detection,
    )

    detection_thread.start()

    return pipeline_cmd

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=55555)
