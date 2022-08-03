docker run  -p 8080:8080 -d --name wlo-classify -v `pwd`/data:/data -v `pwd`/src:/src wlo-cuda:2.0 /usr/bin/python3 /src/webservice.py /data/model-wlo-classification

