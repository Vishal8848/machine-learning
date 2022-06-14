# E7 - Handwritten Digit Recognition : Support Vector Machine

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC

train = pd.read_csv("./E71.csv")
test = pd.read_csv("./E72.csv")

train_data = train["label"].astype("category").value_counts()
print(train_data)

four = train.iloc[3, 1:]

four = four.values.reshape(28, 28)

plt.imshow(four, cmap = "gray")
plt.title("Digit 4")

X = train.drop(columns = "label")
y = train["label"]

test = test / 255.0
X = X / 255.0

print("test  : ", test.shape)
print("train : ", X.shape)

X_scaled = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2, random_state = 10)

linearModel = SVC(kernel = "linear")
linearModel.fit(X_train, y_train)
result = linearModel.predict(X_test)

print("Linear Accuracy: ", accuracy_score(y_true = y_test, y_pred = result))
print(confusion_matrix(y_true = y_test, y_pred = result))

nonLinearModel = SVC(kernel = "rbf")
nonLinearModel.fit(X_train, y_train)
result = nonLinearModel.predict(X_test)

print("Linear Accuracy: ", accuracy_score(y_true = y_test, y_pred = result))
print(confusion_matrix(y_true = y_test, y_pred = result))

plt.show()