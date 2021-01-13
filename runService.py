docker run  -p 8080:8080 -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/webservice.py /data/wirlernenonline.oeh.h5 /data/wirlernenonline.oeh.npy  /data/wirlernenonline.oeh.pickle 

