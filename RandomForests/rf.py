from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def rf(X_train, X_test, y_train, y_test,title):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    return "random forest accuracy for '{0}': {1}".format(title, acc_rf)
