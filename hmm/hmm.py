import pandas as pd
import numpy as np

data = pd.read_csv('hmm/Coding3_HMM_Data.csv')

W = data['Z'].values
X = data['X'].values

A = np.array(((0.54, 0.46), (0.49, 0.51)))
B = np.array(((0.16, 0.26, 0.58), (0.25, 0.28, 0.47)))

initial_distribution = np.array((0.5, 0.5))
