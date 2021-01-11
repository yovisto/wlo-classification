FROM nvidia/cuda:11.0-base

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install unzip python3 python3-pip
RUN apt-get -y install libcublas-11-0  libcufft-11-0 libcurand-11-0 libcusolver-11-0 libcusparse-11-0 libcudnn8

COPY ./requirements.txt /var/code/requirements.txt
WORKDIR /var/code
RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt
