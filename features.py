# TODO: add lots more of these!

from collections import defaultdict

def featurize(review):
	# TODO: make this cleaner
	featurized_review = {}
	for k, v in bag_of_words(review).items():
		featurized_review[k] = v
	for k, v in bigrams(review).items():
		featurized_review[k] = v
	return featurized_review

def bag_of_words(review):
	featurized_review = defaultdict(int)
	for word in review['text'].lower().split():
		featurized_review[word] += 1
	return featurized_review

def bigrams(review):
	featurized_review = defaultdict(int)
	split_review = review['text'].lower().split()
	for i in range(len(split_review) - 1):
		featurized_review[split_review[i] + '_' + split_review[i+1]] += 1
	return featurized_review

