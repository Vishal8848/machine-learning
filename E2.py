# E2 - Spam Email Filter : Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("./E2.csv")
df.groupby("Category").describe()
df["Spam"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)

X_train, X_test, y_train, y_test = train_test_split(df["Message"], df["Spam"])

CV = CountVectorizer()
X_train_count = CV.fit_transform(X_train.values)

model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = [
    "Welcome to Sunday Samayal",
    "Big Offers, 50% Offer on Discount"
]

emails_count = CV.transform(emails)

result = pd.DataFrame(emails, columns=["Message"])

result["Spam"] = model.predict(emails_count)

print(result)