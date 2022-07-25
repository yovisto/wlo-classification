# -*- coding: utf-8 -*-
import json, os, codecs, sys

path = sys.argv[1]
if not os.path.isfile(path):
    print("File '" + path + "' does not exits.")
    sys.exit(1)

textkeys = ["cclom:title", "cm:title", "cm:name", "cclom:general_description", "cm:description"]
kwkeys = ["cclom:general_keyword"]
csv = open(path.replace('.json','.csv'), 'w')

def getText(props):
    text = ""
    for k in textkeys:
        if k in props.keys():
            val = props[k]
            if isinstance(val, list):
                val = " ".join(val)
            text = text + " " + val
    if kwkeys[0] in props.keys():
        text = text + " " + " ".join(props[kwkeys[0]])
    return text.replace('"','').replace('\r',' ').replace('\n',' ')

def valid(json_data):
    if "ccm:collection_io_reference" in json_data.get("_source", None).get("aspects"):
        return False
    # filter collections
    if json_data.get("_source", None).get("type") != "ccm:io":
        return False

    # filter archived and other data
    if json_data.get("_source", None).get("nodeRef").get("storeRef").get("protocol") != "workspace":
        return False

    if json_data.get("_source", None).get("properties").get("cclom:format") == "application/zip":
        return False

    if json_data.get("_source", None).get("properties").get("cclom:title") == None:
        return False

    if json_data.get("_source", None).get("owner") == "WLO-Upload":
        return False

    if json_data.get("_source", None).get("properties").get("cm:edu_metadataset") != "mds_oeh":
        return False
    
    return True

with open(path) as f:
    for line in f:   
        jline=json.loads(line)
        id = jline['_source']['nodeRef']['id']
        props = jline['_source']['properties']   
        if valid(jline) and 'ccm:taxonid' in props.keys():
            disciplines = set(props['ccm:taxonid'])            
            text = getText(props)            
            for disci in disciplines:
                dis = disci.replace('http://w3id.org/openeduhub/vocabs/discipline/','').replace('https://w3id.org/openeduhub/vocabs/discipline/','')
                l = '"' + dis + '","' + text.strip() + '"\n'
                csv.write(l);
csv.close()
