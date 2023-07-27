#! /usr/bin/bash
curl -X POST -H 'Content-Type: application/json' -d '{"url": "http://192.168.2.154/live"}' http://127.0.0.1:55555/pipeline
