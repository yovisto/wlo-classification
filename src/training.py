# -*- coding: utf-8 -*-
import os, sys, re, nltk, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("german")
from transformers import BertTokenizer, TFBertModel
from tqdm.auto import tqdm

dataFile = sys.argv[1]
if not os.path.isfile(dataFile):
    print("File '" + dataFile + "' does not exits.")
    sys.exit(1)

print ("Datafile:", dataFile)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
print("Num GPUs available: ", len(gpus))

#### LOAD DISCIPLINES
disciplines=pd.read_csv("../data/disciplines.csv", sep=';', dtype=str, header=None)
disciplines.columns = ['discipline', 'label']
disciplinesDict={}
for i in disciplines.values:
    disciplinesDict[i[0]]=i[1]
#print (disciplinesDict)

#def disciplineToLabel(text):
##    if text in disciplinesDict.keys():    
#        return disciplinesDict[text]
#    else:
#        return text

blacklist=[]

### LOAD AND PREPROCESS THE DATASET
df = pd.read_csv(dataFile,sep=',', dtype=str, header=None)
df.columns = ['discipline', 'text']
df=df.drop_duplicates()
for b in blacklist:
    df.drop(df[df['discipline'] == b].index, inplace=True)

#samples=1000
#df=df[:samples]

print("Number of samples:" ,len(df))

#df['discipline'] = df['discipline'].apply(disciplineToLabel)

# merge classess
MAPPINGS={'28002':'120','3801':'380','niederdeutsch':'120','04014':'020', '450':'160','04013':'700','400':'900'}
# DaZ, Zahlen, Algebra, Niederdeutsch, Arbeitssicherheit, Philosophie, Wirtschaft und Verwaltung, Mediendidaktik

GARBAGE = ['20003','020','48005','260','04006','50001','64018','340','900','440','44007','04012','640','12002','700','72001','44099'] 
#GARBAGE = []
#Alt-Griechisch 20003
#Arbeitslehre 020
#Gesellschaftskunde 48005
#Gesundheit 260
#'04006': 'Ernährung_und_Hauswirtschaft'
#'50001': 'Hauswirtschaft'
#Nachhaltigkeit 64018
#Interkulturelle_Bildung 340
#Medienbildung 900
#Pädagogik 440
#Sozialpädagogik 44007
#Textiltechnik_und_Bekleidung 04012
#Umweltschuztz 640
#'12002': 'Darstellendes_Spiel'
#'700': 'Wirtschaftskunde'


MAPPINGSD = {}
for k in MAPPINGS:
    #MAPPINGSD[disciplinesDict[k]]=disciplinesDict[MAPPINGS[k]]
    MAPPINGSD[k]= MAPPINGS[k]
#print (MAPPINGSD)

GARBAGED=[]
for k in GARBAGE:
    GARBAGED.append(k) 

# cleanup classes
MIN_NUM=50
for v, c in df.discipline.value_counts().iteritems():
    if c<MIN_NUM or v in GARBAGED:
        MAPPINGSD[v]='other'
#print (MAPPINGSD)    

for k in MAPPINGSD.keys():
    df = df.replace(k, MAPPINGSD[k])


df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}_\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zäöüß ]')
STOPWORDS = set(stopwords.words('german')).union(set(stopwords.words('english'))).union(set(['https','http','lernen','wwwyoutubecom','video','videos','erklärt','einfach','nachhilfe','bitly','online','ordne','mehr','a','hilfe','amznto','wwwfacebookcom','zahlen','b','schule','kostenlos','c','facebook','klasse','unterricht','finden','de','richtigen','themen','fragen','gibt','studium','richtig','richtige','wissen','onlinenachhilfe','finde','schüler','learn','uni','teil','e','weitere','co','aufgaben','twittercom','bild','verben','einzelnen','wwwinstagramcom','berechnen','youtube','twitter','media','lernvideo','quiz','abitur','schnell','thema','free','zeit','website','angaben','erklärvideo','social','bestandteile','mal','top','findest','tet','beispiel','spaß','immer','urhebern','zwei','beim','viele','lizenzbedingungen','seite','kurze','besser','begriffe','infos','la','bzw','plattform','nachhilfeunterricht','lernhilfe','nachhilfelehrer','wurde','onlinehilfe','wer','onlinelehrer','findet','wwwtutoryde','kürze','ordnen','dokument','onlineunterricht','umsonst','world','us','merkhilfe','bereitstellung','schoolseasy','kanal','kostenlose','instagram','schülernachhilfe']))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in STOPWORDS)
    return text

#print (df['text'][:5])
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '', regex=True)
#print (df['text'][:5])


#### TOKENIZE AND CLEAN TEXT
tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")

def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['text'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256, 
            truncation=True, 
            padding='max_length', 
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks


X_input_ids = np.zeros((len(df), 256))
X_attn_masks = np.zeros((len(df), 256))

X_input_ids, X_attn_masks = generate_training_data(df, X_input_ids, X_attn_masks, tokenizer)

labels = pd.get_dummies(df['discipline'])
ds = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))

Y = pd.get_dummies(df['discipline']).values

def DatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, labels

ds = ds.map(DatasetMapFunction)

BATCH_SIZE=64
dataset = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) 
DATASET_SIZE=len(df)//BATCH_SIZE
train_size = int(0.8 * DATASET_SIZE)
val_size = int(0.2 * DATASET_SIZE)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

#### DEFINE THE MODEL

model = TFBertModel.from_pretrained("deepset/gbert-base")
input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1] 
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(len(df['discipline'].value_counts()), activation='softmax', name='output_layer')(intermediate_layer)
model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=1e-6, decay=1e-7)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optim, loss=loss_func, metrics=[acc])

###### DO THE TRAINING
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,    
)

#### SAVE THE MODEL, LABELS AND TOKENIZER
model.save(dataFile.replace('.csv','-model'))

class_names = pd.get_dummies(df['discipline']).columns.values
np.save(dataFile.replace('.csv','_class_names.npy'), class_names)

print ("We are done!")
