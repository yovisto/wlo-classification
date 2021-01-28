# -*- coding: utf-8 -*-
import os, sys, pickle, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


class Prediction:
	
	#should be the same as used for training
	MAX_SEQUENCE_LENGTH=500

	tokenizer, model, class_names = None, None, None

	def __init__(self, modelFile, labelFile, tokenizerFile):
		### We need the same tokenizer as in the training script!!
		self.tokenizer = pickle.load(open(tokenizerFile, 'rb'))
		self.model = tf.keras.models.load_model(modelFile)
		self.class_names = np.load(labelFile, allow_pickle=True)

	REPLACE_BY_SPACE_RE = re.compile('[/(){}_\[\]\|@,;]')
	BAD_SYMBOLS_RE = re.compile('[^0-9a-zäöüß #+_]')
	STOPWORDS = set(stopwords.words('german')).union(set(stopwords.words('english'))).union(set(['https','http','lernen','wwwyoutubecom','video','videos','erklärt','einfach','nachhilfe','bitly','online','ordne','mehr','a','hilfe','amznto','wwwfacebookcom','zahlen','b','schule','kostenlos','c','facebook','klasse','unterricht','finden','de','richtigen','themen','fragen','gibt','studium','richtig','richtige','wissen','onlinenachhilfe','finde','schüler','learn','uni','teil','e','weitere','co','aufgaben','twittercom','bild','verben','einzelnen','wwwinstagramcom','berechnen','youtube','twitter','media','lernvideo','quiz','abitur','schnell','thema','free','zeit','website','angaben','erklärvideo','social','bestandteile','mal','top','findest','tet','beispiel','spaß','immer','urhebern','zwei','beim','viele','lizenzbedingungen','seite','kurze','besser','begriffe','infos','la','bzw','plattform','nachhilfeunterricht','lernhilfe','nachhilfelehrer','wurde','onlinehilfe','wer','onlinelehrer','findet','wwwtutoryde','kürze','ordnen','dokument','onlineunterricht','umsonst','world','us','merkhilfe','bereitstellung','schoolseasy','kanal','kostenlose','instagram','schülernachhilfe']))

	def clean_text(self, text):
		text = text.lower()
		text = self.REPLACE_BY_SPACE_RE.sub(' ', text) 
		text = self.BAD_SYMBOLS_RE.sub('', text) 
		text = ' '.join(word for word in text.split() if word not in self.STOPWORDS)
		return text


	def run(self, text):
		text = self.clean_text(text)
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
		#print (r)
		d1 = r[1][1]-r[0][1]
		d2 = r[2][1]-r[1][1]
		#print (d1,d2)
		if d1>t and d2 <t:
			r = r[1:2]
		if d1<t and d2 > t:
			r = [r[2]]
		f = []
		for i in r:
			if i[1]>m and not i[0]=='0':
				f.append(i)
		#print (f)
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



