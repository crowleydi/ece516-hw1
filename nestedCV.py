# Class for implementing your classifier for SciKit Learn.
# Initial code taken from:
#   http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
from sklearn.base import BaseEstimator, ClassifierMixin

class BoxClassifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    # Initialize by passing the parameters of your model:
    def __init__(self, NumOfPCAcomponents, C, gamma):
        """
        Called when initializing the classifier.
        """
        self.C_ = C
        self.gamma_ = gamma
        self.NumOfPCAcomponents_ = NumOfPCAcomponents

    # Please add PCA and SVM in this template.
    # Make sure to use the given parameters.
    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        """
        ... call .fit() for PCA and .fit for SVC()
        return self

    def predict(self, X):
        """
        Predicts the classification of a single box of pixels.
        """
        ... apply PCA and .predict() for SVC()
        ... Deterimine Class_result for SVC.
        return(Class_result)

    def score(self, X, y):
        """
         Returns the classification accuracy assuming
         balanced datasets (half in each category).
        """
        ... Applies predition on X, and compares the results
        ... against the actual values in y.
        return(accuracy)

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
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])

        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return np.array(outer_scores)


from sklearn.model_selection import ParameterGrid, GroupKFold
# You need to carefully initialize X and y as described previously.
# Furthermore, inner_cv and outer_cv must be initialized using GroupKFold.
# In the parameter_grid, you will need to pass SVM parameters and the
# number of PCA components.
parameter_grid = ParameterGrid(DictionaryOfValues)
scores = nested_cv(X, y, groups, inner_cv, outer_cv, Classifier, parameter_grid)
print("Cross-validation scores: {}".format(scores))
