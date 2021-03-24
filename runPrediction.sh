

docker run -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/predict.py /data/wirlernenonline.oeh3.h5 /data/wirlernenonline.oeh3.npy  /data/wirlernenonline.oeh3.pickle "$1"