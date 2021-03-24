docker run  -p 8080:8080 -d --name wlo-classify -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/webservice.py /data/wirlernenonline.oeh3.h5 /data/wirlernenonline.oeh3.npy  /data/wirlernenonline.oeh3.pickle 

