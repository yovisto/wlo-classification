# -*- coding: utf-8 -*-
import os, sys, re, nltk, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords

dataFile = sys.argv[1]
if not os.path.isfile(dataFile):
    print("File '" + dataFile + "' does not exits.")
    sys.exit(1)

print("Num GPUs available: ", len(tf.config.experimental.list_physical_devices('GPU')))

### LOAD AND PREPROCESS THE DATASET
df = pd.read_csv(dataFile,sep=',')
df.columns = ['discipline', 'text']
#print(df['discipline'].value_counts())

df = df.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}_\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-zäöüß ]')
STOPWORDS = set(stopwords.words('german')).union(set(stopwords.words('english')))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

#print (df['text'][:5])
df['text'] = df['text'].apply(clean_text)
df['text'] = df['text'].str.replace('\d+', '')
#print (df['text'][:5])


#### TOKENIZE AND CLEAN TEXT
# The maximum number of words to be used. (most frequent)
MAX_DICT_SIZE = 200000
# Max number of words in each text.
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
EMBEDDING_DIM = 30

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(MAX_DICT_SIZE, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(tf.keras.layers.SpatialDropout1D(0.2))
model.add(tf.keras.layers.LSTM(100))
model.add(tf.keras.layers.Dense(61, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


###### DO THE TRAINING
EPOCHS = 10
BATCH_SIZE = 128

history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=0.3,callbacks=[
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
print(metrics.classification_report(Y_test, yyy, target_names=df['discipline'].unique()) )

print ("We are done!")
