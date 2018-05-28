# TODO: add lots more of these!
import nltk

from collections import defaultdict
from nltk.stem import *

def featurize(review):
	featurized_review = defaultdict(int)
	bag_of_words(featurized_review, review)
	bigrams(featurized_review, review)
	stars(featurized_review, review)
	stem(featurized_review, review)
	pos_tag(featurized_review, review)
	return featurized_review

def bag_of_words(featurized_review, review):
	for word in review['text'].lower().split():
		featurized_review[word] += 1

def bigrams(featurized_review, review):
	split_review = review['text'].lower().split()
	for i in range(len(split_review) - 1):
		featurized_review[split_review[i] + '_' + split_review[i+1]] += 1

def stars(featurized_review, review):
	featurized_review['FEATURE_STARS'] == review['stars']

def stem(featurized_review, review):
	stemmer = PorterStemmer()
	for word in review['text'].lower().split():
		featurized_review[stemmer.stem(word)] += 1

def pos_tag(featurized_review, review):
	for word, tag in nltk.pos_tag(nltk.word_tokenize(review['text'])):
		featurized_review[word + '_' + tag] += 1