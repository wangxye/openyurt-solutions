# openyurt-solutions

## onvif
```sh
$ git clone https://github.com/dlstreamer/dlstreamer.git

$ cd dlstreamer

$ git am <the patch changing the dockerfile>

build container image
$ docker buildx build -t dlstreamer -f docker/binary/ubuntu/dlstreamer.Dockerfile --platform=linux/amd64 --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}

to test:
start the pipeline server:
$ docker run --rm -p 55555:55555 dlstreamer:latest

on client side:
$ curl -X POST -H 'Content-Type: application/json' -d '{"src": "filesrc", "url": "bbc-fish.mp4", "model": "horizontal-text-detection-0001.xml", "dev": "CPU"}' http://localhost:55555/pipeline

```
