cd data
if [ ! -f "wirlernenonline.oeh3.csv" ]; then
	unzip wirlernenonline.oeh3.csv.zip 
fi
cd ..

docker run --runtime=nvidia -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/training4.py /data/wirlernenonline.oeh4.csv

#docker run --runtime=nvidia -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/training.py /data/wirlernenonline.oeh.csv