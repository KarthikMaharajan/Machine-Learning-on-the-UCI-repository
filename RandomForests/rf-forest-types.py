""" random forest for forest types dataset """
import pandas as pd
import numpy as np
from sklearn import cross_validation
from rf import rf # custom function (in rf.py) for random forests

import os
os.chdir("/Users/sandy/work/UF/ML/Project/code/RandomForests")

""" training.csv """
train = pd.read_csv("../../datasets/ForestTypes/training.csv")
features = train.columns[1:]
X_train = train[features]
X_train = X_train.astype(float)
y_train = train['class']

""" testing.csv """
test = pd.read_csv("../../datasets/ForestTypes/testing.csv")
features = test.columns[1:]
X_test = test[features]
X_test = X_test.astype(float)
y_test = test['class']

print rf(X_train, X_test, y_train, y_test,"Forest types")
