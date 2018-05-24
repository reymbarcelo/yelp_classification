# TODO: add lots more of these!

from collections import defaultdict


def bag_of_words(raw_review):
	featurized_review = defaultdict(int)
	for word in raw_review['text'].split():
		featurized_review[word] += 1
	return featurized_review