# ExamCram — PCCSL508 Machine Learning Lab, last-minute edition

Same lab cycle as the numbered solutions in this repo, rewritten to be **crammed right before the exam**.
Every file keeps the simplest possible sklearn flow: **load → split → (scale) → fit → predict → metrics → plot**,
with a short header at the top stating the question and the 3–4 things to remember.

Covers questions: **01, 02, 03, 05, 07, 11, 13, 19, 20**.

Run from the repo root (paths point at `Datasets/`):

```bash
python ExamCram/01.py
```

---

## The universal skeleton (memorize this once)

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()               # for regression/SVM/MLP, NOT for trees
X_train = sc.fit_transform(X_train) # fit on TRAIN only
X_test  = sc.transform(X_test)      # just transform TEST
```

## Question-by-question crib sheet

| Q | Topic | Model(s) | Dataset | One-line memory hook |
|---|-------|----------|---------|----------------------|
| 01 | Linear regression 1 var | `SGDRegressor` vs `LinearRegression` | housing | GD needs `learning_rate='constant', eta0`; normal equation needs nothing |
| 02 | Polynomial regression | `PolynomialFeatures` + `LinearRegression` | autompg | `fit_transform` train, `transform` test; still "linear" in expanded features |
| 03 | Regularization | `RidgeCV` (L2), `LassoCV` (L1) | diabetes | CV picks `alpha_`; L1 can zero coefficients |
| 05 | MLE vs MAP (multinomial) | hand formulas | 20newsgroups | MLE `Nw/N`; MAP `(Nw+a-1)/(N+V(a-1))`; a=1 → MAP = MLE |
| 07 | Naïve Bayes text | `MultinomialNB` vs `BernoulliNB` | 20newsgroups | counts for Multinomial (`binary=False`), 0/1 presence for Bernoulli (`binary=True`) |
| 11 | Linear SVM | `SVC(kernel='linear')` | iris | boundary `w.x+b=0`, margins at `w.x+b=±1`; support vectors define everything |
| 13 | MLP architectures | `MLPClassifier(hidden_layer_sizes)` | wine_quality | `(20,10)` = 2 hidden layers; scale once outside the loop |
| 19 | Ensembles | `BaggingClassifier`, `AdaBoostClassifier` | titanic | impute + encode first; bagging cuts variance, boosting cuts bias |
| 20 | Bias–variance | poly degrees 1–5, train vs val MSE | housing | low degree = high bias, high degree = exploding val error |

## Formulas worth writing on paper

- **MLE:** `P(w) = count(w) / N`
- **MAP (Dirichlet α):** `P(w) = (count(w) + α − 1) / (N + V·(α − 1))`
- **SVM lines:** `x2 = −(w1·x + b)/w2`, then `+1` and `−1` variants of `b` for the margins
- **Metrics:** MSE = mean((y − ŷ)²), R² = 1 − SS_res/SS_tot, F1 = 2·P·R/(P+R) (use `average='weighted'` for multiclass)
