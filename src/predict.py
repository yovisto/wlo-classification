# -*- coding: utf-8 -*-
import os, sys, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


class Prediction:
	
	#should be the same as used for training
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
		r = sorted(result, key=lambda x: x[1])[-3:]
		r.reverse()
		t = 0.1
		m=0.3
		print (r)
		d1 = r[1][0]-r[0][0]
		d2 = r[2][0]-r[1][0]
		#print (d1,d2)
		if d1>t and d2 <t:
		r = r[1:2]
		if d1<t and d2 > t:
			r = [r[2]]
		f = []
		for i in r:
			if i[0]>m:
			f.append(i)
		print (f)
		return f


if __name__ == '__main__':	

	modelFile = sys.argv[1]
	labelFile = sys.argv[2]
	tokenizerFile = sys.argv[3]
	text = sys.argv[4]

	print ("Predicting: '" + text + "'")

	r = Prediction(modelFile, labelFile, tokenizerFile)
	for r in r.run(text):
		print (r)



