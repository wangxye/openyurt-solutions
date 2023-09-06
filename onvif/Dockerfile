# FROM python:3.9-alpine
FROM ubuntu:20.04

ARG DLS_HOME=/home/dlstreamer
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR ${DLS_HOME}
RUN apt-get update && apt-get install -y python3.8 python3.8-dev
RUN apt-get install -y git python3-pip

ADD . .

ARG bar=
RUN cd ${DLS_HOME} \
    && git clone https://github.com/wangxye/openyurt-solutions.git
RUN python3 -m pip install -U pip
RUN apt install -y libgl1-mesa-glx 
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-dev

RUN pip3 install --no-cache-dir -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple


EXPOSE 55555

CMD ["python3", "/home/dlstreamer/openyurt-solutions/onvif/pipeline.py"]
