from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

df = pd.read_csv(r".\Datasets\20newsgroups.csv")

texts = df['text'].fillna("")

vectorizer = CountVectorizer(stop_words="english", max_features=1000)

X = vectorizer.fit_transform(texts)

vocabulary = vectorizer.get_feature_names_out() 

word_counts = np.asarray(X.sum(axis=0)).flatten()

total_words = word_counts.sum() 

print(f"Vocabulary size: {len(vocabulary)} \nTotal words: {total_words}")

mle = word_counts / total_words 

alphas = [0.5, 1, 2]
map_results = {} 
for alpha in alphas:
    map_results[alpha] = (word_counts + alpha) / (total_words + alpha * len(vocabulary))

compare_words = ["computer", "windows", "god", "space", "game"]

print("Comparison of words: ")
print(f"{'Word':<15}{'MLE':<12}{'α=0.5':<12}{'α=1':<12}{'α=2':<12}")

for word in compare_words:
    if word in vocabulary:
        idx = np.where(vocabulary == word)[0][0]
        
        print(
            f"{word:<15}"
            f"{mle[idx]:<12.6f}"
            f"{map_results[0.5][idx]:<12.6f}"
            f"{map_results[1][idx]:<12.6f}"
            f"{map_results[2][idx]:<12.6f}"
        )

top = np.argsort(mle)[-10:][::-1]

print("\nTop 10 Words (MLE)")
print("-"*30)

for i in top:
    print(vocabulary[i], ":", round(mle[i], 6))