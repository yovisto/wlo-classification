# -*- coding: utf-8 -*-
import os, sys, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


modelFile = sys.argv[1]
labelFile = sys.argv[2]
example = [sys.argv[3]]

print ("Predicting: '" + example[0] + "'")

### We need the same tokenizer as in the training script!
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

# LOAD MODEL AND LABELS
model = tf.keras.models.load_model(modelFile)
class_names = np.load(labelFile, allow_pickle=True)

# PREDICTION
seq = tokenizer.texts_to_sequences(example)
padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
result = []
for i in range(len(pred[0])):
   result.append( [pred[0][i], class_names[i]])
rev = sorted(result)[-3:]
rev.reverse()
print (rev)
