import numpy as np
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class BoxClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, C, gamma):
		self.numPCA_ = 120
		self.C_ = C
		self.gamma_ = gamma
	
	def fit(self, X ,y):
		self.pca_ = PCA(n_components=self.numPCA_,
			svd_solver='randomized').fit(X)
		X_pca = self.pca_.transform(X)
		self.svc_ = SVC(kernel='rbf', class_weight='balanced').fit(X_pca, y)
		return self
	
	def predict(self, X):
		return self.svc_.predict(self.pca_.transform(X))
	
	def score(self, X, y):
		y_predict = self.predict(X)
		correct = 0
		for i in range(0,len(y)):
			if y[i] == y_predict[i]:
				correct = correct + 1
		accuracy=correct/len(y)
		return accuracy

def nested_cv(X, y, groups, inner_cv, outer_cv, Classifier, parameter_grid):
	"""
	Uses nested cross-validation to optimize and exhaustively evaluate
	the performance of a given classifier. The original code was taken from
	Chapter 5 of Introduction to Machine Learning with Python. However, it
	has been modified.

	Input parameters:
		X, y, groups: describe one set of boxes grouped by image number.

	Output:
		The function returns the scores from the outer loop.
	"""
	outer_scores = []
	# for each split of the data in the outer cross-validation
	# (split method returns indices of training and test parts)
	#
	for training_samples, test_samples in outer_cv.split(X, y, groups):
		# find best parameter using inner cross-validation
		best_parms = {}
		best_score = -np.inf
		# iterate over parameters
		for parameters in parameter_grid:
			# accumulate score over inner splits
			cv_scores = []
			# iterate over inner cross-validation
			for inner_train, inner_test in inner_cv.split(
				X[training_samples], y[training_samples],
				groups[training_samples]):
				# build classifier given parameters and training data
				clf = Classifier(**parameters)
				clf.fit(X[inner_train], y[inner_train])

				# evaluate on inner test set
				score = clf.score(X[inner_test], y[inner_test])
				cv_scores.append(score)

			# compute mean score over inner folds
			# for a single combination of parameters.
			mean_score = np.mean(cv_scores)
			if mean_score > best_score:
				# if better than so far, remember parameters
				best_score = mean_score
				best_params = parameters

		# Build classifier on best parameters using outer training set
		# This is done over all parameters evaluated through a single
		# outer fold and all inner folds.
		print("Best params:")
		print(best_params)
		clf = Classifier(**best_params)
		clf.fit(X[training_samples], y[training_samples])
		# evaluate
		outer_scores.append(clf.score(X[test_samples], y[test_samples]))
	return np.array(outer_scores), best_params

param_dict = {
	'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	#'numPCA': [100, 150, 200],
	'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
	}
Scales=[1, 1, 2, 2]
AspectRatios=[1, 2, 1, 2]

for scale, aspectRatio in zip(Scales, AspectRatios):
	print("Loading data...")
	X = np.load("X_{}_{}.npy".format(scale,aspectRatio))
	y = np.load("y_{}_{}.npy".format(scale,aspectRatio))
	groups = np.load("groups_{}_{}.npy".format(scale,aspectRatio))

	inner_cv = KFold(n_splits=5, shuffle=True, random_state=5)
	outer_cv = KFold(n_splits=5, shuffle=True, random_state=6)

	parameter_grid = ParameterGrid(param_dict)
	print("Finding best model for scale/aspect {}/{}...".format(scale,aspectRatio))
	scores,params = nested_cv(X, y, groups, inner_cv, outer_cv, BoxClassifier,
		parameter_grid)
	print("Cross-validation scores: {}".format(scores))
	print("Training final classifier....")
	clf = BoxClassifier(params)
	clf.fit(X, y)
	print("Saving classifier...")
	dump(clf, "clf_{}_{}.joblib".format(scale,aspectRatio))
