# TODO: try out different models!

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

def chosen_model():
	return SGDClassifier(max_iter=5)