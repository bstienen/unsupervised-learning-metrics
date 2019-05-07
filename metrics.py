"""
UNSUPERVISED LEARNING METRICS
Author:     Bob Stienen
License:    MIT License
Source:     http://www.github.com/bstienen/AUMVC

Implementation of the Area under the Mass-Volume Curve algorithm as described by
- Stephan Clemencon and Jeremie Jakubowicz, Scoring anomalies: a M-estimation
  formulation approach. 2013-04

Implementation is inspired by
   https://github.com/albertcthomas/anomaly_tuning
"""

import warnings
import numpy as np
from scipy.misc import comb
from sklearn.metrics import auc


def aumvc(scoring_function,
          X_test,
          N_mc=100000,
          N_levelsets=100,
          normalise=True):
    """ Calculate the area under the mass-volume curve for an anomaly detection
    function or algorithm

    This function uses monte carlo sampling in the parameter space box spanned
    by the provided test data in order to estimate the level set of the
    scoring function. For higher dimensionalities the amount of sampled data
    points would yield this algorithm intractable. In these cases the use of
    the `aumvc_hd` function is advised instead.

    Parameters
    ----------
    scoring_function: function
        Function that takes datapoints as numpy.ndarray (nPoints, nFeatures)
        and returns an anomaly score. This score should be in range [0,1], where
        1 indicates the point not being an anomaly (and 0 that the point *is* an
        anomaly).
    X_test: numpy.ndarray of shape (nPoints, nFeatures)
        Datapoints used for testing the algorithm.
    N_mc: int (default: 100,000)
        Number of datapoints to sample in the parameter space to estimate the
        level sets of the scoring function.
    N_levelsets: int (default: 100)
        Number of level sets to evaluate.
    normalise: bool (default: True)
        Indicates if output scores of the scoring_function should be normalised
        before calculating the mass-volume curve. """

    # Get ranges for the test data
    mins = np.amin(X_test, axis=0)
    maxs = np.amax(X_test, axis=0)

    # Generate uniform MC data
    U = np.random.rand(N_mc, len(mins))*(maxs-mins)+mins

    # Calculate volume of total cube
    vol_tot_cube = np.prod(maxs-mins)

    # Score test and MC data
    score_U = scoring_function(U)
    score_test = scoring_function(X_test)

    # Do normalising if needed
    if normalise:
        minimum = min(np.amin(score_U), np.amin(score_test))
        maximum = max(np.amax(score_U), np.amax(score_test))
        score_U = (score_U - minimum) / (maximum - minimum)
        score_test = (score_test - minimum) / (maximum - minimum)

    # Calculate alphas to use
    alphas = np.linspace(0, 1, N_levelsets)

    # Compute offsets
    offsets = np.percentile(score_test, 100 * (1 - alphas))

    # Compute volumes of associated level sets
    volumes = (np.array([np.mean(score_U >= offset) for offset in offsets]) *
              vol_tot_cube)

    # Calculating area under the curve
    area = auc(alphas, volumes)

    # Return area and curve variables
    return (area, alphas, volumes)


def aumvc_hd(scoring_function_generator,
             X_train,
             X_test,
             N_selected_dim=5,
             N_iterations=100,
             N_mc=100000,
             N_levelsets=1000,
             normalise=True):
    """ Calculate the area under the mass-volume curve for an anomaly detection
    function or algorithm working in high-dimensional parameter spaces

    The curse of dimensionality is avoided by taking the average over multiple
    AUMVC values for randomly selected subspaces of the parameter space under
    consideration. The AUMVCs are calculated using the `aumvc` function above.
    As this requires a retraining of the scoring function for each random
    subspace, the `aumvc_hd` function does not take a scoring function as input,
    but rather a generator of scoring functions. This function should take
    the training data as input and return a scoring function (see description
    of `aumvc` for requirements of this function).

    Parameters
    ----------
    scoring_function_generator: function
        Function that takes training datapoints as numpy.ndarray of shape
        (nPoints, nFeatures) and returns a scoring function. See description of
        `aumvc` function for requirements on the scoring function.
    X_train: numpy.ndarray of shape (nPoints, nFeatures)
        Data points for which randomly selected subspaces are passed to the
        scoring function generator for creation of the scoring function.
    X_test: numpy.ndarray of shape (nPoints, nFeatures)
        Data points used for testing the algorithm. Number of data points does
        not have to match the number of training points, but the number of
        features *does* have to match.
    N_selected_dim: int (default=5)
        Number of dimensions selected for the random subspace generation. This
        number should be equal to or smaller than the number of features in
        the testing data.
    N_iterations: int (default=100)
        Number of random subspaces have to be evaluated. A warning will be
        raised if this number is higher than the total number of unique
        combinations that can be randomly selected from the provided parameter
        space.
    N_mc: int (default=100,000)
        Number of datapoints to sample in the parameter space to estimate the
        level sets of the scoring function.
    N_levelsets: int (default=100)
        Number of level sets to evaluate.
    normalise: bool (default: True)
        Indicates if output scores of the scoring_function should be normalised
        before calculating the mass-volume curve. """


    # Check if N_selected_dim <= dim(X_test)
    data_dim = X_test.shape[1]
    if data_dim < N_selected_dim:
        raise Exception("The number of dimensions to select in each iteration "
                        "is larger than the number of dimensions in the "
                        "provided data.")

    # Check if the dimensionality of training data matches the dimensionality
    # of the testing data
    if X_train.shape[1] != data_dim:
        raise Exception("The number of features in the training data does not "
                        "match the number of features in the testing data.")

    # Check if the number of unique random subspaces is significantly larger
    # (i.e. > a factor of 2) than the requested number of iterations
    N_unique = comb(data_dim, N_selected_dim)
    if N_unique < 2*N_selected_dim:
        warnings.warn("The number of unique combinations of the dimensions of "
                      "the input space is smaller than the number of "
                      "dimensions to select in each iterations.")

    # Initialise final AUMVC variable
    area_hd = 0

    # Run over each iteration
    for iteration in range(N_iterations):

        # Make feature subselection
        features = np.random.choice(data_dim, N_selected_dim, replace=False)
        X_selection = X_test[:, features]
        X_train_selection = X_train[:, features]

        # Train scoring function
        scoring_function = scoring_function_generator(X_train_selection)

        # Calculate area under curve and collect it in final variable
        area, _, _ = aumvc(scoring_function,
                           X_selection,
                           N_mc,
                           N_levelsets,
                           normalise)
        area_hd += area

    # Return mean area
    return area_hd / N_iterations
