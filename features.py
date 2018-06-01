# TODO: add lots more of these!
import nltk
import sys

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import *

def featurize(review):
	featurized_review = defaultdict(int)
	# bag_of_words(featurized_review, review)
	# bigrams(featurized_review, review)
	stars(featurized_review, review)
	# stem(featurized_review, review)
	# pos_tag(featurized_review, review)
	preprocessed_bag_of_words(featurized_review, review)
	service_vs_food(featurized_review, review)
	return featurized_review

def bag_of_words(featurized_review, review):
	for word in review['text'].lower().split():
		featurized_review[word] += 1

def bigrams(featurized_review, review):
	split_review = review['text'].lower().split()
	for i in range(len(split_review) - 1):
		featurized_review[split_review[i] + '_' + split_review[i+1]] += 1

def stars(featurized_review, review):
	featurized_review['FEATURE_STARS'] = review['stars']

def stem(featurized_review, review):
	stemmer = PorterStemmer()
	for word in review['text'].lower().split():
		featurized_review[stemmer.stem(word)] += 1

def pos_tag(featurized_review, review):
	for word, tag in nltk.pos_tag(nltk.word_tokenize(review['text'])):
		featurized_review[word + '_' + tag] += 1

# Combines bag_of_words, stem, and pos_tag into one funcion.
def preprocessed_bag_of_words(featurized_review, review):
	stemmer = PorterStemmer()
	stop = set(stopwords.words('english'))
	for word, tag in nltk.pos_tag(nltk.word_tokenize(review['text'])):
		if word in stop:
			continue
		featurized_review[stemmer.stem(word) + '_' + tag] += 1	

def service_vs_food(featurized_review, review):
	stemmer = PorterStemmer()
	service_words = set([])
	food_words = set([])
	service_word_count = 0
	food_word_count = 0
	with open('service_words.txt') as s:
		for line in s:
			service_words.add(stemmer.stem(line.lower()[:-1]))
	with open('food_labels.txt') as f:
		for line in f:
			food_words.add(stemmer.stem(line.lower()[:-1]))
	for original_word in nltk.word_tokenize(review['text']):
		stemmed_word = stemmer.stem(original_word)
		if stemmed_word in service_words:
			service_word_count += 1
			featurized_review['SERVICE_' + stemmed_word] += 1
		elif stemmed_word in food_words:
			food_word_count += 1
			featurized_review['FOOD_' + stemmed_word] += 1
	food_service_ratio = float(food_word_count / (service_word_count + sys.float_info.epsilon))
	featurized_review['FOOD_SERVICE_RATIO'] = food_service_ratio



















