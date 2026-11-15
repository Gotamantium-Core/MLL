# Question 07 - Naive Bayes text classification (20 Newsgroups)
# MultinomialNB vs BernoulliNB on bag-of-words features, accuracy + weighted F1.
# CRAM POINTS:
#   * MultinomialNB <- word counts (binary=False); BernoulliNB <- 0/1 word presence (binary=True).
#   * CountVectorizer(stop_words='english', max_features=5000) limits vocab size.
#   * Compare with accuracy_score + f1_score(average='weighted').

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB,MultinomialNB

df = pd.read_csv("Datasets/20newsgroups.csv")

X = df["text"].fillna("")
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

# Bernoulli NB: binary word-presence features
nb = CountVectorizer(binary=True,stop_words='english',max_features=2000)

X_train_nb = nb.fit_transform(X_train)
X_test_nb = nb.transform(X_test)

nb_model = BernoulliNB()
nb_model.fit(X_train_nb,y_train)
y_pred_nb = nb_model.predict(X_test_nb)

# Multinomial NB: raw word counts
mb = CountVectorizer(binary=False,stop_words='english')
X_train_mb = mb.fit_transform(X_train)
X_test_mb = mb.transform(X_test)

mb_model = MultinomialNB()
mb_model.fit(X_train_mb,y_train)
y_pred_mb = mb_model.predict(X_test_mb)

print("Performance Metrics - Bernoulli Naive Bayes")
print(f"Accuracy Score = {accuracy_score(y_test,y_pred_nb)}")
print(f"F1 Score = {f1_score(y_test,y_pred_nb,average='weighted')}")

print("Performance Metrics - Multinomial Naive Bayes")
print(f"Accuracy Score = {accuracy_score(y_test,y_pred_mb)}")
print(f"F1 Score = {f1_score(y_test,y_pred_mb,average='weighted')}")
