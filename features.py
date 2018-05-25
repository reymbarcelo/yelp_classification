# TODO: add lots more of these!

from collections import defaultdict

def featurize(review):
	# TODO: make this cleaner
	featurized_review = defaultdict(int)
	bag_of_words(featurized_review, review)
	bigrams(featurized_review, review)
	stars(featurized_review, review)
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

