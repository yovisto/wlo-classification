
# retrieve and unpack WLO dataset

cd data
wget https://github.com/openeduhub/oeh-wlo-data-dump/raw/main/data/wirlernenonline.oeh.json.gz 
gunzip wirlernenonline.oeh.json.gz
cd ..

# convert to csv

 docker run -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/prepareData.py /data/wirlernenonline.oeh.json