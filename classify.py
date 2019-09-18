import numpy as np
from sklean.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid, KFold

class BoxClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, numPCA, C, gamma):
		self.numPCA_ = numPCA
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
		for i in len(y):
			if y[i] == y_predict[i]:
				correct = correct + 1
		accuracy=correct/len(y)
		return accuracy

def nested_cv(X, y, groups, inner_cv, outer_cv, Classifier, parameter_grid):
	outer_scores = []
	for training_samples, test_samples in outer_cv.split(X,y,groups):
		# find best parameters
		best_params = {}
		best_score = -1.0
		# iterate over parameters
		for parameters in parameter_grid:
			# accumulate score over inner split
			cv_scores = []
			for inner_train, inner_test in inner_cv.split(
				X[training_samples],y[training_samples],
				groups[training_samples]):

				clf = Classifier(**parameters)
				clf.fit(X[inner_train], y[inner_train])
				score = clf.score(X[inner_test],y[inner_test])

				cv_scores.append(score)

			mean_score = np.mean(cv_scores)
			if mean_score > best_score
				best_score = mean_score
				best_params = parameters

		# build classifier with the best parameters
		clf = Classifier(**best_params)
		clf.fit(X[training_samples], y[training_samples])

		score = clf.score(X[test_samples],y[test_samples])
		outer_scores.append(score)

	return np.array(outer_scores)


param_dict = {
	'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	'numPCA': [100, 150, 200],
	'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
	}
Scales=[1, 1, 2, 2]
AspectRatios=[1, 2, 1, 2]

for scale, aspectRatio in zip(Scales, AspectRatios):
	X = np.load("X_{}_{}.npy".format(scale,aspectRatio))
	y = np.load("y_{}_{}.npy".format(scale,aspectRatio))

	inner_cv = KFold(n_splits=5, shuffle=True, random_state=5)
	outer_cv = KFold(n_splits=5, shuffle=True, random_state=6)

	parameter_grid = ParameterGrid(param_dict)
	scores = nested_cv(X, y, groups, inner_cv, outer_cv, BoxClassifier,
		parameter_grid)
	print("Cross-validation scores: {}".format(scores))
