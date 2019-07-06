import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

import metrics


# Generate data
data, _ = make_classification(n_samples = 2000,
                              n_features = 10,
                              n_informative = 10,
                              n_redundant = 0,
                              n_repeated = 0,
                              n_classes = 2,
                              n_clusters_per_class = 2,
                              hypercube = False,
                              shuffle = True)
train, test = train_test_split(data, train_size=0.5)

# Train OneClassSVM and make predictions on the test set
clf = OneClassSVM()
clf.fit(train)
pred = clf.score_samples(test)

# Plot predictions
plt.figure(figsize=(6,6))
plt.scatter(test[:,0], test[:,1], c=pred, s=10)
plt.show()

# Run AUMVC
area, alphas, volumes = metrics.aumvc(scoring_function = clf.score_samples,
                                      X_test = test,
                                      N_mc = int(1e5),
                                      N_levelsets = 100,
                                      normalise = True)

# Plot mass-volume curve
plt.plot(alphas, volumes)
plt.xlabel("Mass")
plt.ylabel("Volume")
plt.show()

# Create scoring function generator
def gen(X):
    clf = IsolationForest(behaviour="new")
    clf.fit(X)
    return clf.score_samples

# Run AUMVC_hd
area = metrics.aumvc_hd(scoring_function_generator = gen,
                        X_train = train,
                        X_test = test,
                        N_selected_dim = 5,
                        N_iterations = 10,
                        N_mc = int(1e5),
                        N_levelsets = 1000,
                        normalise = True)

print("AUMVC_hd area: {}".format(area))
