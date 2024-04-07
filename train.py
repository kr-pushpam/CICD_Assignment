#Note - this verison of code works for Case 1- Both of your actions pass
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# import pickle
# import numpy as np

# df = pd.read_csv("data/train.csv")
# X = df.drop(columns=['Disease']).to_numpy()
# y = df['Disease'].to_numpy()
# labels = np.sort(np.unique(y))
# y = np.array([np.where(labels == x) for x in y]).flatten()

# model = LogisticRegression().fit(X, y)

# with open("model.pkl", 'wb') as f:
#     pickle.dump(model, f)


# Case 1- Both of your actions pass - passed 02:45 - tran triggred by pull - and test on completion of train

#################################################################################################################

# Test Case 2 - Both of your actions fail ( both needs to run and fail)
# Changes -  Train.yml and train.py


import pandas as pd
from sklearn.linear_model import LogisticRegression
# Intentional error: missing import for pickle
# import pickle
import numpy as np

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Intentional error: LogisticRegression is incorrectly instantiated
model = LogisticRegression(wrong_parameter=True).fit(X, y)

# Intentional error: 'pickle' is not imported, so this will fail
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)

