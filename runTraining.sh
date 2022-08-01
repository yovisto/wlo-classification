cd data
if [ ! -f "wirlernenonline.oeh.csv" ]; then
	unzip wirlernenonline.oeh.csv.zip 
fi
cd ..

docker run --runtime=nvidia -v `pwd`/data:/data -v `pwd`/src:/src wlo-cuda:2.0 /usr/bin/python3 /src/training.py /data/wirlernenonline.oeh.csv