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

dataFile = sys.argv[1]
if not os.path.isfile(dataFile):
    print("File '" + dataFile + "' does not exits.")
    sys.exit(1)

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
disciplines=pd.read_csv("/data/disciplines.csv", sep=';', dtype=str, header=None)
disciplines.columns = ['discipline', 'label']
disciplinesDict={}
for i in disciplines.values:
    disciplinesDict[i[0]]=i[1]
#print (disciplinesDict)

def disciplineToLabel(text):
    if text in disciplinesDict.keys():    
        return disciplinesDict[text]
    else:
        return text


### LOAD AND PREPROCESS THE DATASET
df = pd.read_csv(dataFile,sep=',')
df.columns = ['discipline', 'text']
#print(df['discipline'].value_counts())

df['discipline'] = df['discipline'].apply(disciplineToLabel)

# merge classess
MAPPINGS={'28002':'120','3801':'380','niederdeutsch':'120','04014':'020', '450':'160','04013':'700','400':'900'}
# DaZ, Zahlen, Algebra, Niederdeutsch, Arbeitssicherheit, Philosophie, Wirtschaft und Verwaltung, Mediendidaktik

GARBAGE = ['20003','020','48005','260','04006','50001','64018','340','900','440','44007','04012','640','12002','700'] 
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
    MAPPINGSD[disciplinesDict[k]]=disciplinesDict[MAPPINGS[k]]
#print (MAPPINGSD)

GARBAGED=[]
for k in GARBAGE:
    GARBAGED.append(disciplinesDict[k]) 

# cleanup classes
MIN_NUM=50
for v, c in df.discipline.value_counts().iteritems():
    if c<MIN_NUM or v in GARBAGED:
        MAPPINGSD[v]='garbage'
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
df['text'] = df['text'].str.replace('\d+', '')
#print (df['text'][:5])


#### TOKENIZE AND CLEAN TEXT
# The maximum number of words to be used. (most frequent)
MAX_DICT_SIZE = 50000
# Max number of words in each text.
# should be the same as used in prediction
MAX_SEQUENCE_LENGTH = 500

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_DICT_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['text'].values)
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['discipline']).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)
print('Shapes of train test split:')
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


#### DEFINE THE MODEL
EMBEDDING_DIM = 50

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(MAX_DICT_SIZE, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(tf.keras.layers.SpatialDropout1D(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


###### DO THE TRAINING
EPOCHS = 20
BATCH_SIZE = 1024

history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=0.1,callbacks=[
tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


#### SAVE THE MODEL, LABELS AND TOKENIZER
model.save(dataFile.replace('.csv','.h5'))

class_names = pd.get_dummies(df['discipline']).columns.values
np.save(dataFile.replace('.csv','.npy'), class_names)

with open(dataFile.replace('.csv','.pickle'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


#### CHECK EVALUATION RESULTS
print("EVALUATION")
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

print("Testing prediction ...")
y_pred = model.predict(X_test)
yyy = np.zeros_like(y_pred)
yyy[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
labels = pd.get_dummies(df['discipline']).columns.values
print(metrics.classification_report(Y_test, yyy, target_names=labels) )

print ("We are done!")
