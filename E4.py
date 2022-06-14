# E4 - Predict Titanic Survival Probability : Decision Tree Classifier

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

titanic = pd.read_csv("./E4.csv")
titanic.drop(["passenger_id", "name", "sibsp", "parch", "ticket", "embarked", "cabin", "boat", "body", "home.dest"], axis = "columns", inplace = True)

input = titanic.drop("survived", axis = "columns")
target = titanic["survived"]

input["sex"] = input["sex"].map({ "male" : 1, "female" : 2 })
input["age"] = input["age"].fillna(input["age"].mean())
input["fare"] = input["fare"].fillna(input["fare"].mean())

X_train, X_test, y_train, y_test = train_test_split(input, target, test_size = 0.2)

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
    return 1 - ( (s_prob * s_prob) * (ns_prob * ns_prob) )

print("Overall Gini Impurity: ")
print(get_gini_impurity(342, 891))

print("Gini Impurity for Men: ")
print(get_gini_impurity(109, 577))

print("Gini Impurity for Women: ")
print(get_gini_impurity(233, 314))