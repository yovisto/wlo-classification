

docker run --runtime=nvidia -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/predict.py /data/wirlernenonline.oeh.h5  /data/wirlernenonline.oeh.npy "$1"