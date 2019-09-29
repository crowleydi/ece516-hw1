import cv2
import numpy as np
from joblib import dump, load
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import argparse

class BoxClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, C, gamma):
		self.numPCA_ = 120
		self.C_ = C
		self.gamma_ = gamma
	
	def fit(self, X ,y, probability=False):
		self.pca_ = PCA(n_components=self.numPCA_,
			svd_solver='randomized').fit(X)
		X_pca = self.pca_.transform(X)
		self.svc_ = SVC(kernel='rbf', C=self.C_, gamma=self.gamma_,
			probability=probability, class_weight='balanced').fit(X_pca, y)
		return self
	
	def predict(self, X):
		return self.svc_.predict(self.pca_.transform(X))
	
	def predict_proba(self, X):
		return self.svc_.predict_proba(self.pca_.transform(X))

	def score(self, X, y):
		return np.sum(y==self.predict(X))/len(y)

# unit box is 120x120
BoxUnitWidth = 120
AnchorPointSpacing = 60
ImgX = 1920
ImgY = 1080
Scales=[1, 1, 2, 2]
AspectRatios=[1, 2, 1, 2]

# Generate all anchor points
def GenerateAnchorPoints():
	# calculate anchor point spacing along x
	nx = int(ImgX/AnchorPointSpacing)
	# calculate anchor point spacing along y
	ny = int(ImgY/AnchorPointSpacing)
	print("nx="+str(nx))
	print("ny="+str(ny))
	# calculate all of the X and Y anchor points
	apxs = np.array(range(0,nx))*(ImgX/nx)+AnchorPointSpacing/2
	apys = np.array(range(0,ny))*(ImgY/ny)+AnchorPointSpacing/2
	# combine all of the x/y combinations into a list
	aps=[]
	for apx in apxs:
		for apy in apys:
			tup = (int(apx),int(apy))
			aps.append(tup)
	return aps

# Generate boxes given the anchor points and
# a given scale and aspect ratio
def GenerateBoxes(anchors, scale, aspectRatio):
	boxes = []
	dx = (scale * BoxUnitWidth)
	dy = (dx * aspectRatio)
	for ap in anchors:
		# "center" of the box
		# the center is calculated such that it is shifted
		# away from the edges so boxes don't go off the edge.
		cx = min(max(dx,ap[0]),ImgX-dx)
		cy = min(max(dy,ap[1]),ImgY-dy)
		# upper left of thebox
		ux = cx-dx/2
		uy = cy-dy/2
		# box tuple contains(upper_left_x, upper_left_y, width, height)
		tup=(int(ux),int(uy),int(dx),int(dy))
		boxes.append(tup)
	return boxes

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
	outer_params = {}
	outer_score = -np.inf
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
			if mean_score > outer_score:
				outer_score = mean_score
				outer_params = parameters

		# Build classifier on best parameters using outer training set
		# This is done over all parameters evaluated through a single
		# outer fold and all inner folds.
		print("Best params:")
		print(best_params)
		clf = Classifier(**best_params)
		clf.fit(X[training_samples], y[training_samples])
		# evaluate
		outer_scores.append(clf.score(X[test_samples], y[test_samples]))
	return np.array(outer_scores), outer_params

def ExtractBox(img, box):
	sub = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
	return sub

parser = argparse.ArgumentParser()
parser.add_argument("--traincv", help="Train using cross-validation",
                    action="store_true")
parser.add_argument("--video", nargs=1, action='store', help="inference the first frame of the specified video file")
args = parser.parse_args()

if args.traincv == True:
	param_dict = {
		'C': [1, 10, 100, 1000, 10000],
		#'numPCA': [100, 150, 200],
		'gamma': [5e-5, 1e-4, 5e-4, 1e-3]
		}

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
		clf = BoxClassifier(**params).fit(X,y,probability=True)
		print("Saving classifier...")
		dump(clf, "clf_{}_{}.joblib".format(scale,aspectRatio))

elif args.video:
	# read the first frame
	print("Reading frame...")
	cap = cv2.VideoCapture(args.video[0])
	ret, frame = cap.read()
	if ret == False:
		print("Could not read video file: {}".format(args.video))
	else:
		# Convert to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		aps = GenerateAnchorPoints()
		for scale, aspectRatio in zip(Scales, AspectRatios):
			# load the appropriate model
			print("Loading model {}/{}...".format(scale,aspectRatio))
			clf = load("clf_{}_{}.joblib".format(scale,aspectRatio))

			# generate the boxes
			boxes = GenerateBoxes(aps, scale, aspectRatio)
			datawidth = boxes[0][2]*boxes[0][3]

			# extract the boxes from the image
			X = np.zeros((len(boxes),datawidth))
			i = 0
			for box in boxes:
				X[i,:] = ExtractBox(gray, box).flatten()
				i = i + 1

			y = clf.predict_proba(X)[:,1]
			print(y)
