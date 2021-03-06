# Classifies reviews based on top N classes.

import json
import models
import os
import sklearn
import sys
import time

from collections import defaultdict
from features import featurize
from random import shuffle
from sklearn.feature_extraction import DictVectorizer

# Some combinations of these params are impossible. If so,
# it *should* exit gracefully, but I can't promise anything :P
NUM_CLASSES = 10
NUM_RELEVANT_CLASSES = 3
NUM_REVIEWS = 1000
PERCENT_TRAIN = 0.75
EPSILON = sys.float_info.epsilon
verbose = (len(sys.argv) > 1)
directory = '../dataset/reviews'

classes = set([]) 					# set(['Nightlife', 'Bars', ...])
review_texts = []					# ['This place is the WORST', ...]
featurized_reviews = []				# [{'review_id': ...}, {}, ...]
labels = []							# [['American', 'Burgers'], ['Italian', ...], ...]
my_models = [None] * NUM_CLASSES	# [SGDClassifier for 'Burgers', SGDClassifier for 'Sushi', ...]
predicted = [[]] * NUM_CLASSES 		# [[1, 0, 1, ...], ...]
actual = [[]] * NUM_CLASSES 		# [[1, 1, 0, ...], ...]

# Generate top NUM_CLASSES classes
i = 0
with open('labels.txt') as labels_file:
	for line in labels_file:
		if(i == NUM_CLASSES):
			break
		classes.add(line[:-1])
		i += 1

# Read in data
review_files = os.listdir(directory)
shuffle(review_files)
i = 0
while len(featurized_reviews) < NUM_REVIEWS:
	filename = directory + '/' + review_files[i]
	i += 1
	if i >= len(review_files):
		exit('There are not enough reviews for the parameters \
			you have selected. Please try again.')
	with open(filename) as review_file:
		for line in review_file:
			try:
				review = json.loads(line)
			except:
				continue
			relevant_classes = classes.intersection(set(review['categories']))
			if len(relevant_classes) < NUM_RELEVANT_CLASSES:
				continue
			review_texts.append(review['text'])
			featurized_reviews.append(featurize(review))
			labels.append(list(relevant_classes))

# Generate train, test data
num_train_reviews = int(PERCENT_TRAIN * len(featurized_reviews))
train_reviews = featurized_reviews[:num_train_reviews]
train_labels = labels[:num_train_reviews]

num_test_reviews = NUM_REVIEWS - num_train_reviews
test_reviews = featurized_reviews[num_train_reviews:]
test_review_texts = review_texts[num_train_reviews:]
test_labels = labels[num_train_reviews:]

# Fit data
v = DictVectorizer(sparse=False)
X_train = v.fit_transform(train_reviews)
X_test = v.transform(test_reviews)

model = sklearn.cluster.KMeans(NUM_CLASSES)
model.fit(X_train)
# TODO: finish this

# Fancy print predictions and add evaluation metrics to a file
numCorrect = 0
total = 0
true_positives = [EPSILON] * NUM_CLASSES
false_positives = [EPSILON] * NUM_CLASSES
false_negatives = [EPSILON] * NUM_CLASSES
for i in range(num_test_reviews):
	review_text = review_texts[i]
	if verbose: 
		print('########REVIEW########')
		print(review_text[:100])
		print(('{:>20s}: {:>10s} {:>10s} {:>10s}').format('Class', 'Predicted', 'Actual', 'Correct?'))
	for j in range(NUM_CLASSES):
		if predicted[j][i] == 1:
			if actual[j][i] == 1:
				true_positives[j] += 1
			else:
				false_positives[j] += 1
		elif actual[j][i] == 1:
			false_negatives[j] += 1
		correct = (predicted[j][i] == actual[j][i])
		if correct:
			numCorrect += 1
		total += 1
		if verbose: print(('{:>20s}: {:>10s} {:>10s} {:>10s}').format(classes[j][:20], \
			'YES' if predicted[j][i] == 1 else 'NO', \
			'YES' if actual[j][i] == 1 else 'NO',
			'+' if correct else ''))
	if verbose and input('Hit ENTER to continue, hit anything else to quit. ') != '': 
		verbose = False
print('#########EVAL#########')
print(('{:>20s}: {:>10s} {:>10s} {:>10s}').format('Class', 'Precision', 'Recall', 'Instances'))
for j in range(NUM_CLASSES):
	print('{:>20s}: {:>1.8f} {:>1.8f} {:>10d}'.format(classes[j][:20], \
			float(true_positives[j] / (true_positives[j] + false_positives[j])), \
			float(true_positives[j] / (true_positives[j] + false_negatives[j])), \
			sum(actual[j])))
print('Accuracy:', float(numCorrect / (total)))







