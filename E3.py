# E3 - Predict Age of Abalone : K Nearest Neighbors

import pandas as pd
import numpy as np
import scipy.stats as ss

abalone = pd.read_csv("./E3.csv")
abalone = abalone.drop("Sex", axis = 1)

# Correlation Matrix
CM = abalone.corr()
print(CM["Rings"])

a = np.array([2, 2])
b = np.array([4, 4])
print(np.linalg.norm(a - b))

X = abalone.drop("Rings", axis = 1)
X = X.values
y = abalone["Rings"]
y = y.values

data_point = np.array([
    0.569552,
    0.446407,
    0.154437,
    1.016849,
    0.439051,
    0.222526,
    0.291208,
])

distances = np.linalg.norm(X - data_point, axis = 1)

# K = 11
nearest_neighbor_ids = distances.argsort()[:11]
print(nearest_neighbor_ids)

nearest_neighbor_rings = y[nearest_neighbor_ids]
print(nearest_neighbor_rings)

print(nearest_neighbor_rings.mean())
print(ss.mode(nearest_neighbor_rings))