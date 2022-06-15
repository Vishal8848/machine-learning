# E9 - Wisconsin Breast Cancer Detection : Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bc = pd.read_csv("./E9.csv")
del bc["id"]

X_train, X_test, y_train, y_test = train_test_split(
    bc.loc[:, bc.columns != "diagnosis"], bc["diagnosis"], 
    stratify = bc["diagnosis"], random_state = 66  )

model = GradientBoostingClassifier(max_depth = 1)
model.fit(X_train, y_train)

print("Gradient Boosting Training Set Accuracy: {}".format(model.score(X_train, y_train)))
print("Gradient Boosting Testing Set Accuracy: {}".format(model.score(X_test, y_test)))
features = [ x for i, x in enumerate(bc.columns) if i != 30 ]

plt.figure(figsize = (10, 5))
plt.barh(range(30), model.feature_importances_, align = "center", color = ["#FF1534"])
plt.yticks(np.arange(30), features)
plt.title("Breast Cancer Gradient Boosting Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.ylim(-1, 30)
plt.show()