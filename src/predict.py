# -*- coding: utf-8 -*-
import os, sys, pickle, re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
np.set_printoptions(suppress=True)

class Prediction:

	tokenizer, model = None, None

	# class names ant its order need to fit to the model
	class_names= ['040', '04003', '060', '080', '100', '120', '160', '20001', '20002', '20004', '20005', '20006', '20007', '20008', '220', '240','28010', '320', '380', '420', '460', '46014', '480', '50005','510', '520', '600', '720', 'other']

	def __init__(self, modelFile):
		### We need the same tokenizer as in the training script!!
		self.tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")		
		self.model = tf.keras.models.load_model(modelFile)		

	REPLACE_BY_SPACE_RE = re.compile('[/(){}_\[\]\|@,;]')
	BAD_SYMBOLS_RE = re.compile('[^0-9a-zäöüß ]')
	STOPWORDS = set(stopwords.words('german')).union(set(stopwords.words('english'))).union(set(['https','http','lernen','wwwyoutubecom','video','videos','erklärt','einfach','nachhilfe','bitly','online','ordne','mehr','a','hilfe','amznto','wwwfacebookcom','zahlen','b','schule','kostenlos','c','facebook','klasse','unterricht','finden','de','richtigen','themen','fragen','gibt','studium','richtig','richtige','wissen','onlinenachhilfe','finde','schüler','learn','uni','teil','e','weitere','co','aufgaben','twittercom','bild','verben','einzelnen','wwwinstagramcom','berechnen','youtube','twitter','media','lernvideo','quiz','abitur','schnell','thema','free','zeit','website','angaben','erklärvideo','social','bestandteile','mal','top','findest','tet','beispiel','spaß','immer','urhebern','zwei','beim','viele','lizenzbedingungen','seite','kurze','besser','begriffe','infos','la','bzw','plattform','nachhilfeunterricht','lernhilfe','nachhilfelehrer','wurde','onlinehilfe','wer','onlinelehrer','findet','wwwtutoryde','kürze','ordnen','dokument','onlineunterricht','umsonst','world','us','merkhilfe','bereitstellung','schoolseasy','kanal','kostenlose','instagram','schülernachhilfe']))

	def clean_text(self, text):
		text = text.lower()
		text = self.REPLACE_BY_SPACE_RE.sub(' ', text)
		text = self.BAD_SYMBOLS_RE.sub('', text)
		text = ' '.join(word for word in text.split() if word not in self.STOPWORDS)
		return text

	def prepare_data(self, input_text):
		input_text = self.clean_text(input_text)
		token = self.tokenizer.encode_plus(
			input_text,
			max_length=256, 
			truncation=True, 
			padding='max_length', 
			add_special_tokens=True,
			return_tensors='tf'
		)
		return {
			'input_ids': tf.cast(token.input_ids, tf.float64),
			'attention_mask': tf.cast(token.attention_mask, tf.float64)
		}

	def make_prediction(self, processed_data):
		probs = self.model.predict(processed_data)[0]
		#print (probs)
		#return self.class_names[np.argmax(probs)]
		return probs

	def run(self, text):
		processed_data = self.prepare_data(text)
		#print (processed_data)
		pred = self.make_prediction(processed_data)
		result=[]
		for i in range(len(pred)):
		   result.append( (self.class_names[i], pred[i].astype(float) ))
		r = sorted(result, key=lambda x: x[1])[-3:]
		r.reverse()
		t = 0.1
		m = 0.2
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
			if i[1]>m: 
				f.append(i)
		result = []
		for e in f:
			if len(f)>1:
				if e[0]!='other':
					result.append(e)
			else:
				result.append(e)
		return result


if __name__ == '__main__':	

	modelFile = sys.argv[1]
	text = sys.argv[2]

	print ("Predicting: '" + text + "'")

	r = Prediction(modelFile)
	for r in r.run(text):
		print (r)



