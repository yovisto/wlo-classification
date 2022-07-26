

docker run -v `pwd`/data:/data -v `pwd`/src:/src wlo-cuda:1.0 /usr/bin/python3 /src/predict.py /data/wirlernenonline.oeh.h5 /data/wirlernenonline.oeh.npy  /data/wirlernenonline.oeh.pickle "$1"