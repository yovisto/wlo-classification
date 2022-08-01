cd data
if [ ! -f "wirlernenonline.oeh.csv" ]; then
	unzip wirlernenonline.oeh.csv.zip 
fi
cd ..

run --runtime=nvidia -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/training.py /data/wirlernenonline.oeh.csv