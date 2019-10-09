import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
from tqdm import tqdm
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import SpectralClustering, AffinityPropagation

import utils
plt.ion()
plt.show()

nb_clusters = 7
X, y_true = make_blobs(n_samples=300, centers=nb_clusters,
                       cluster_std=.80, random_state=0)
plt.title(f'Ground truth simulated data : {nb_clusters} clusters')
plt.scatter(X[:, 0], X[:, 1], s=50, c = y_true);

print(utils.internalValidation(X, y_true))
