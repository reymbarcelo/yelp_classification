# TODO: try out different models!

# import tensorflow as tf
import random

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

class ExampleModel():
	# Creates the model
	def __init__(self):
		return

	# Params:
	# 	X_train: 2d feature vector
	# 	Y_train: 1d label vector
	# Returns: None
	def fit(self, X_train, Y_train):
		return

	# Params:
	# 	X_test: 2d feature vector
	# Returns: 
	#	Y_test: 1d label vector
	def predict(self, X_test):
		return []

########################SKLEARN#MODELS########################

class AlwaysNoModel(ExampleModel):
	def __init__(self):
		return

	def fit(self, X_train, Y_train):
		return

	def predict(self, X_test):
		return [0 for X in X_test]

class RandomModel(ExampleModel):
	def __init__(self):
		return

	def fit(self, X_train, Y_train):
		return

	def predict(self, X_test):
		return [(0 if random.uniform(0, 1) < 0.5 else 1) for X in X_test]

class SGDModel(ExampleModel):
	def __init__(self):
		self.model = SGDClassifier(max_iter=20)

	def fit(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

class LinearSVCModel(ExampleModel):
	def __init__(self):
		self.model = LinearSVC(max_iter=20)

	def fit(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

class NonlinearSVCModel(ExampleModel):
	def __init__(self):
		self.model = SVC(max_iter=20)

	def fit(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

class NeighborsModel():
	def __init__(self):
		self.model = KNeighborsClassifier()

	def fit(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

class NeuralModel():
	def __init__(self):
		self.model = MLPClassifier()

	def fit(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

class DTModel():
	def __init__(self):
		self.model = DecisionTreeClassifier()

	def fit(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

########################TENSORFLOW#MODELS########################

# TODO: fix this
# class TFModel(ExampleModel):
# 	def __init__(self):
# 		self.model = tf.estimator.DNNClassifier(
# 			hidden_units=[256, 32],
# 			feature_columns=[tf.feature_column.numeric_column('x', shape=[28, 28])]
# 		)

# 	def fit(self, X_train, Y_train):
# 		self.model.train(X_train)

# 	def predict(self, X_test):
# 		return self.model.predict(X_test)

########################PYTORCH#MODELS########################

# TODO: finish this
# class TorchModel(ExampleModel):
# 	def __init__(self):
# 		self.model = torch.nn.Sequential(
# 			torch.nn.Linear(input_num_units, hidden_num_units),
# 			torch.nn.ReLU(),
# 			torch.nn.Linear(hidden_num_units, output_num_units),
# 		)
# 		self.loss_fn = torch.nn.CrossEntropyLoss()
# 		self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 		self.epochs = 5

# 	def fit(self, X_train, Y_train):
# 		return

# 	def predict(self, X_test):
# 		return []






