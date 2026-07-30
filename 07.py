import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score


df = pd.read_csv(r".\Datasets\20newsgroups.csv")

X = df["text"].fillna("") 
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer_multi = CountVectorizer(stop_words="english", max_features=5000)

X_train_multi = vectorizer_multi.fit_transform(X_train)
X_test_multi = vectorizer_multi.transform(X_test)

multi = MultinomialNB() 
multi.fit(X_train_multi, y_train)

y_pred_multi = multi.predict(X_test_multi)

print("Multinomial Naive Bayes")
print(f"Accuracy: {accuracy_score(y_test, y_pred_multi)}")
print(f"F1 Score: {f1_score(y_test, y_pred_multi, average="weighted")}")


vectorizer_binary = CountVectorizer(stop_words="english", max_features=5000, binary=True)

X_train_binary = vectorizer_binary.fit_transform(X_train)
X_test_binary = vectorizer_binary.transform(X_test)

bernoulli = BernoulliNB() 

bernoulli.fit(X_train_binary, y_train)
y_pred_binary = bernoulli.predict(X_test_binary)

print("\nBernoulli Naive Bayes")
print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
print(f"F1 Score: {f1_score(y_test, y_pred_binary, average="weighted")}")
