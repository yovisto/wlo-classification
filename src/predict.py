# -*- coding: utf-8 -*-
import os, sys, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


class Prediction:
	
	MAX_SEQUENCE_LENGTH=500

	tokenizer, model, class_names = None, None, None

	def __init__(self, modelFile, labelFile, tokenizerFile):
		### We need the same tokenizer as in the training script!!
		self.tokenizer = pickle.load(open(tokenizerFile, 'rb'))
		self.model = tf.keras.models.load_model(modelFile)
		self.class_names = np.load(labelFile, allow_pickle=True)

	def run(self, text):
		seq = self.tokenizer.texts_to_sequences([text])
		padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
		pred = self.model.predict(padded)
		result = []
		for i in range(len(pred[0])):
		   result.append( (self.class_names[i], pred[0][i].astype(float) ))
		rev = sorted(result, key=lambda x: x[1])[-3:]
		rev.reverse()
		return rev


if __name__ == '__main__':	

	modelFile = sys.argv[1]
	labelFile = sys.argv[2]
	tokenizerFile = sys.argv[3]
	text = sys.argv[4]

	print ("Predicting: '" + text + "'")

	r = Prediction(modelFile, labelFile, tokenizerFile)
	for r in r.run(text):
		print (r)



