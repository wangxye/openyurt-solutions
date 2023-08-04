#! /usr/bin/bash
curl -X POST -H 'Content-Type: application/json' -d '{"src": "filesrc", "url": "bbc-fish.mp4", "model": "horizontal-text-detection-0001.xml", "dev": "CPU"}' http://10.238.158.178:55555/pipeline
