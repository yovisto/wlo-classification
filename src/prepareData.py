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
    return text.replace('"','').replace('\n',' ')

with open(path) as f:
    for line in f:
        jline=json.loads(line)
        id = jline['_source']['nodeRef']['id']
        props = jline['_source']['properties']
        if 'ccm:taxonid' in props.keys():
            disciplines = set(props['ccm:taxonid'])
            text = getText(props)
            for discipline in disciplines:
                dis = discipline.replace('http://w3id.org/openeduhub/vocabs/discipline/','').replace('https://w3id.org/openeduhub/vocabs/discipline/','')
                l = dis + ',"' + text + '"\n'
                csv.write(l);
csv.close()
