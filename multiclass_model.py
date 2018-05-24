# Classifies reviews based on top N classes.
# Uses SKLearn.SGDClassifier as a model and bag of words as feature extraction.

import features
import json
import os

from collections import defaultdict
from random import shuffle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier


NUM_CLASSES = 5
NUM_REVIEWS = 10
PERCENT_TRAIN = 0.9
verbose = True

directory = '../dataset/reviews'

classes = set([]) 					# set(['Nightlife', 'Bars', ...])
original_reviews = []				# ['This place is the WORST', ...]
featurized_reviews = []				# [{'review_id': ...}, {}, ...]
labels = []							# [['American', 'Burgers'], ['Italian', ...], ...]
models = [None] * NUM_CLASSES		# [SGDClassifier for 'Burgers', SGDClassifier for 'Sushi', ...]
predictions = [0] * NUM_CLASSES

def featurize(raw_review):
	return features.bag_of_words(raw_review)

# Generate top NUM_CLASSES classes
i = 0
with open('labels.txt') as labels_file:
	for line in labels_file:
		if(i == NUM_CLASSES):
			break
		classes.add(line[:-1])
		i += 1

# Read in data
review_data = os.listdir(directory)
shuffle(review_data)
i = 0
while len(featurized_reviews) < NUM_REVIEWS:
	filename = directory + '/' + review_data[i]
	i += 1
	with open(filename) as review_file:
		for line in review_file:
			try:
				raw_review = json.loads(line)
			except:
				continue
			relevant_classes = classes.intersection(set(raw_review['categories']))
			if len(relevant_classes) == 0:
				continue
			original_reviews.append(raw_review['text'])
			featurized_reviews.append(featurize(raw_review))
			labels.append(list(relevant_classes))

# Generate train, test data
num_train_reviews = int(PERCENT_TRAIN * len(featurized_reviews))
train_reviews = featurized_reviews[:num_train_reviews]
train_labels = labels[:num_train_reviews]
test_reviews = featurized_reviews[num_train_reviews:]
test_original_reviews = original_reviews[num_train_reviews:]
test_labels = labels[num_train_reviews:]

# Fit data
v = DictVectorizer(sparse=False)
X_train = v.fit_transform(train_reviews)
X_test = v.transform(test_reviews)

# Create N models
classes = list(classes)
for i in range(NUM_CLASSES):
	# Eg. if labels = [('Bars', 'Burgers'), ('Burgers', 'Sushi'), ('Sushi')]
	# 	  and classes[i] = 'Burgers', then
	#     	class_specific_labels = [1, 1, 0]
	class_specific_labels = [1 if classes[i] in set(label_list) else 0 for label_list in train_labels]
	models[i] = SGDClassifier()
	models[i].fit(X_train, class_specific_labels)
	predictions[i] = models[i].predict(X_test)
	if verbose:
		for original_review, prediction, label in zip(test_original_reviews, predictions[i], test_labels):
			print(original_review)
			print('Predicted:', 'YES' if prediction == 1 else 'NO', classes[i])
			print('Actual:', 'YES' if classes[i] in test_labels else 'NO', classes[i])
			input()








print('Made it to the end!')





