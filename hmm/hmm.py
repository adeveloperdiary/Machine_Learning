import pandas as pd
import numpy as np
import copy

data = pd.read_csv('hmm/Coding3_HMM_Data.csv')

W = data['Z'].values
X = data['X'].values

a = np.array(((0.54, 0.46), (0.49, 0.51)))
b = np.array(((0.16, 0.26, 0.58), (0.25, 0.28, 0.47)))

initial_distribution = np.array((0.5, 0.5))

alpha=np.copy(initial_distribution)

for t in range(X.shape[0]):
    alpha_t_1=copy.deepcopy(alpha)
    for j in range(a.shape[0]):
        alpha[j]=alpha_t_1.T.dot(a[j,:])*b[j,X[t]-1]