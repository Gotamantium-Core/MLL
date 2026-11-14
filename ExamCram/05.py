# Question 05 - MLE vs MAP for multinomial parameters (20 Newsgroups)
# Word probabilities from raw counts: MLE = N_w/N, MAP with Dirichlet(a) = (N_w+a-1)/(N+V(a-1)).
# CRAM POINTS:
#   * MLE formula: count / total. MAP formula: (count + alpha - 1) / (total + V*(alpha-1)).
#   * alpha=1 => MAP equals MLE (uniform prior); larger alpha => probabilities pulled toward uniform.
#   * Compare MLE vs MAP(a=1,2,5,10) for a few words + top-10 words per estimate.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("Datasets/20newsgroups.csv")

texts = df["text"].fillna("")
vectorizer = CountVectorizer(stop_words='english',max_features=1000)
X = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()

word_count = np.asarray(X.sum(axis=0)).flatten()
N = word_count.sum()
V = len(vocab)
print(f"Vocabulary size: {V}")
print(f"Total words: {N}")

mle = word_count / N
alphas = [1,2,5,10]
map_results = {}
for alpha in alphas:
    map_est = (word_count + alpha - 1)/(N + V * (alpha - 1))
    map_results[alpha] = map_est

compare_words = ["computer","windows","god","space","game"]
print("---------------------------------")
print("Word | MLE | MAP Probs for alphas:", alphas)
#arrays for plotting
mle_p = []
map_1 = []
map_2 = []
map_5 = []
map_10 = []
for word in compare_words:
    if word in vocab:
        idx = np.where(vocab == word)[0][0]  # vectorized lookup, no manual loop needed

        map_prob = []

        for alpha in alphas:
            map_prob.append(float(map_results[alpha][idx]))

        # Print everything separated by spaces
        print(word, mle[idx], map_prob)

        mle_p.append(mle[idx])
        map_1.append(map_prob[0])
        map_2.append(map_prob[1])
        map_5.append(map_prob[2])
        map_10.append(map_prob[3])


def print_top_words(prob,title):
    print("----------------------")
    print(title)
    word_probs = []
    for i in range(len(prob)):
        word_probs.append((prob[i],vocab[i]))

    word_probs.sort(reverse = True)
    word_probs = word_probs[:10]
    for w in word_probs:
        print(w[1],w[0])

print_top_words(mle,"Top 10 probabilities in MLE")
for alpha in alphas:
    print_top_words(map_results[alpha],f"Top 10 probabilities in MAP with alpha {alpha}")

#plotting

data = {
    'MLE': mle_p,
    'MAP a=1': map_1,
    'MAP a=2': map_2,
    'MAP a=5': map_5,
    'MAP a=10': map_10
}

df_p = pd.DataFrame(data,index=["computer","windows","god","space","game"])
df_p.plot.bar()
plt.show()
