#Note - this verison of code works 
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

model = GradientBoostingClassifier(learning_rate=10, max_depth=1).fit(X,y)

with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
