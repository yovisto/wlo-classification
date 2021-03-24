cd data
unzip wirlernenonline.oeh3.csv.zip 
cd ..

docker run --runtime=nvidia -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/training3.py /data/wirlernenonline.oeh3.csv