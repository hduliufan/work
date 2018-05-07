import pandas as pd
import numpy as np

df = pd.read_csv('trainingdatabases.csv', header=None)
print(np.shape(df))
print(df.tail())
