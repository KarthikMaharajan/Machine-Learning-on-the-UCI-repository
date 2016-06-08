""" random forest for breast cancer dataset """
import pandas as pd
import numpy as np
from sklearn import cross_validation
from rf import rf # custom function (in rf.py) for random forests

import os
os.chdir("/Users/sandy/work/UF/ML/Project/code/RandomForests")

""" breast-cancer-wisconsin.data """
data = pd.read_csv("../../datasets/breast-cancer-wisconsin/breast-cancer-wisconsin.data",header=None)
data = data.replace('?',np.nan).astype(float,raise_on_error=False).dropna(how='any') # removing missing values
num_columns = len(data.columns)
features = data.columns[1:-1]
X = data[features]
y = data[num_columns-1]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)

print rf(X_train, X_test, y_train, y_test,"Wisconsin Breast Cancer Database")

""" wdbc.data """
data = pd.read_csv("../../datasets/breast-cancer-wisconsin/wdbc.data",header=None)
data = data.astype(float,raise_on_error=False).dropna(how='any') # removing missing values
features = data.columns[2:]
X = data[features]
y = data[1]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)
print rf(X_train, X_test, y_train, y_test,"Wisconsin Diagnostic Breast Cancer")

""" wpbc.data """
data = pd.read_csv("../../datasets/breast-cancer-wisconsin/wpbc.data",header=None)
data = data.replace('?',np.nan).astype(float,raise_on_error=False).dropna(how='any') # removing missing values
features = data.columns[3:]
X = data[features]
y = data[1]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)
print rf(X_train, X_test, y_train, y_test,"Wisconsin Prognostic Breast Cancer")
