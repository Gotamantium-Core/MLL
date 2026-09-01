from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r".\Datasets\20newsgroups.csv")

texts = df["text"].fillna("")

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1000
)

X = vectorizer.fit_transform(texts)

vocabulary = vectorizer.get_feature_names_out()

word_counts = np.asarray(X.sum(axis=0)).flatten()

N = word_counts.sum()
V = len(vocabulary)

print(f"Vocabulary size: {V}")
print(f"Total words: {N}")

mle = word_counts / N

alphas = [1, 2, 5, 10]

map_results = {}

for alpha in alphas:

    # Symmetric Dirichlet(alpha) prior
    map_estimate = (
        word_counts + alpha - 1
    ) / (
        N + V * (alpha - 1)
    )

    map_results[alpha] = map_estimate



compare_words = [
    "computer",
    "windows",
    "god",
    "space",
    "game"
]

print("\nComparison of word probabilities")
print("-" * 70)

header = f"{'Word':<15}{'MLE':<12}"

for alpha in alphas:
    header += f"MAP α={alpha:<8}"

print(header)

for word in compare_words:

    if word in vocabulary:

        idx = np.where(vocabulary == word)[0][0]

        row = f"{word:<15}{mle[idx]:<12.6f}"

        for alpha in alphas:
            row += f"{map_results[alpha][idx]:<12.6f}"

        print(row)


def print_top_words(probabilities, title):

    top = np.argsort(probabilities)[-10:][::-1]

    print(f"\n{title}")
    print("-" * 40)

    for i in top:
        print(f"{vocabulary[i]:<15} {probabilities[i]:.6f}")


print_top_words(mle, "Top 10 Words - MLE")

for alpha in alphas:
    print_top_words(
        map_results[alpha],
        f"Top 10 Words - MAP (α={alpha})"
    )


plot_words = [
    word for word in compare_words
    if word in vocabulary
]

plot_data = {
    "MLE": [
        mle[np.where(vocabulary == word)[0][0]]
        for word in plot_words
    ]
}

for alpha in alphas:
    plot_data[f"MAP α={alpha}"] = [
        map_results[alpha][np.where(vocabulary == word)[0][0]]
        for word in plot_words
    ]

comparison_df = pd.DataFrame(
    plot_data,
    index=plot_words
)

comparison_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("Estimated Probability")
plt.xlabel("Word")
plt.title("MLE vs MAP with Different Dirichlet Priors")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
