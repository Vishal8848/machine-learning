# E4 - Predict Titanic Survival Probability : Decision Tree Classifier

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

titanic = pd.read_csv("./E4.csv")
titanic.drop(["passenger_id", "sibsp", "name", "parch", "cabin", "boat", "body", "ticket", "embarked", "home.dest"],  axis = "columns", inplace = True)

inputs = titanic.drop("survived", axis = "columns")
target = titanic["survived"]

inputs["sex"] = inputs["sex"].map({ "male" : 1, "female" : 2 })
inputs["age"] = inputs["age"].fillna(inputs["age"].mean())
inputs["fare"] = inputs["fare"].fillna(inputs["fare"].mean())

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size = 0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

test_data = {
    "pclass" : [3, 3],
    "sex" : [2, 1],
    "age" : [29.0, 38.0],
    "fare" : [7.7333, 8.6625]
}

result = pd.DataFrame(test_data, columns = ["pclass", "sex", "age", "fare"])
result["survived"] = model.predict(result)
print(result)

def get_gini_impurity(survived_count, total_count):
    s_prob = survived_count / total_count
    ns_prob = (1 - s_prob)
    return 1 - ((s_prob * s_prob) + (ns_prob * ns_prob))

print("Overall Gini Impurity: ")
print(get_gini_impurity(313, 850))

print("Gini Impurity for Men: ")
print(get_gini_impurity(103, 551))

print("Gini Impurity for Women: ")
print(get_gini_impurity(210, 299))