# PCCSL508 - Machine Learning Lab


#### Note
For as many programs as I can, I extract the values from the dataset in `sklearn.datasets` into a csv file and do the required problem by reading from the file. However for some programs (fashion, online retail datasets etc.) the csv is too big to upload directly here. As such, I'd recommend downloading them from [here](kaggle.com) directly.


## Lab Cycle 
[Taken directly from the official KTU syllabus]

1. Implement linear regression with one variable on the California Housing dataset to predict
housing prices based on a single feature (e.g., the average number of rooms per dwelling).
Tasks:
- Load and preprocess the datase.
- Implement linear regression using both gradient descent and the normal equation.
- Evaluate the model performance using metrics such as Mean Squared Error
(MSE) and R-squared.
- Visualize the fitted line along with the data points.

2. Implement polynomial regression on the Auto MPG dataset to predict miles per gallon
(MPG) based on engine displacement. Compare polynomial regression results with linear
regression.
Tasks:
- Load and preprocess the dataset.
- Implement polynomial regression of varying degrees.
- Compare the polynomial regression models with linear regression using metrics
such as MSE and R-squared.
- Visualize the polynomial fit.

3. Implement Ridge and Lasso regression on the Diabetes dataset. Compare the performance
of these regularized models with standard linear regression.
Tasks:
- Load and preprocess the dataset.
- Implement Ridge and Lasso regression.
- Tune hyperparameters using cross-validation.
- Compare performance metrics (MSE, R-squared) with standard linear regression.

4. Estimate the parameters of a logistic regression model using MLE and MAP on the Breast
Cancer Wisconsin dataset. Compare the results and discuss the effects of regularization.
Tasks:
- Load and preprocess the dataset.
- Implement logistic regression with MLE.
- Apply MAP estimation with different regularization priors (L1 and L2
regularization).
- Compare the performance and parameter estimates with MLE and MAP.

5. Use MLE and MAP to estimate the parameters of a multinomial distribution on the 20
Newsgroups dataset. Explore the impact of different priors on the estimation.
Tasks:
- Load and preprocess the dataset.
- Implement MLE for multinomial distribution parameter estimation.
- Apply MAP estimation with various priors (e.g., Dirichlet priors).
- Compare results and evaluate the effect of different priors.

6. Implement a logistic regression model to predict the likelihood of a disease using the Pima
Indians Diabetes dataset. Compare the performance with and without feature scaling.
Tasks:
- Load and preprocess the Pima Indians Diabetes dataset.
- Implement logistic regression for binary classification.
- Evaluate model performance with and without feature scaling.
- Analyze metrics such as accuracy, precision, recall, and F1-score.

7. Implement a Naïve Bayes classifier to categorize text documents into topics using the 20
Newsgroups dataset. Compare the performance of Multinomial Naïve Bayes with
Bernoulli Naïve Bayes.
Tasks:
- Load and preprocess the 20 Newsgroups dataset.
- Implement Multinomial Naïve Bayes and Bernoulli Naïve Bayes classifiers.
- Evaluate and compare the performance of both models using metrics such as
accuracy and F1-score.
- Discuss the strengths and weaknesses of each Naïve Bayes variant for text
classification.

8. Implement the K-Nearest Neighbors (KNN) algorithm for image classification using the
Fashion MNIST dataset. Experiment with different values of K and analyze their impact
on model performance.
Tasks:
- Load and preprocess the Fashion MNIST dataset.
- Implement KNN for multi-class classification.
- Experiment with different values of K and evaluate performance.
- Discuss the impact of different K values on model accuracy and computational
efficiency.

9. Implement a Decision Tree classifier using the ID3 algorithm to segment customers based
on their purchasing behavior using the Online Retail dataset. Analyze the tree structure
and discuss the feature importance.
Tasks:
- Load and preprocess the Online Retail dataset.
- Implement Decision Tree using the ID3 algorithm.
- Visualize the decision tree and analyze feature importance.
- Discuss how the tree structure helps in understanding customer behavior.

10. Implement and compare Logistic Regression and Decision Trees on the Adult Income
dataset for predicting income levels. Evaluate both models based on performance metrics
and interpretability.
Tasks:
- Load and preprocess the Adult Income dataset.
- Implement both Logistic Regression and Decision Trees.
- Compare the models based on metrics such as accuracy, precision, recall, and F1-
score.
- Discuss the interpretability of both models and their suitability for the dataset.

11. Implement a Linear Support Vector Machine (SVM) to classify the Iris dataset. Visualize
the decision boundary and discuss how the margin is determined.
Tasks:
- Load and preprocess the Iris dataset.
- Implement a Linear SVM for binary classification (e.g., classify Setosa vs. Non-
Setosa).
- Visualize the decision boundary and margin.
- Discuss the concept of the margin and how it influences classification.

12. Implement and compare the performance of SVM classifiers with linear, polynomial, and
RBF kernels on the Fashion MNIST dataset. Analyze the advantages and disadvantages of
each kernel type.
Tasks:
- Load and preprocess the Fashion MNIST dataset.
- Implement SVM with linear, polynomial, and RBF kernels.
- Compare the classification performance for each kernel.
- Discuss the strengths and weaknesses of each kernel type.


13. Implement and train a Multilayer Feed-Forward Network (MLP) on the Wine Quality
dataset. Experiment with different numbers of hidden layers and neurons, and discuss how
these choices affect the network’s performance.
Tasks:
- Load and preprocess the Wine Quality dataset.
- Design and implement an MLP with varying architectures (different hidden layers
and neurons).
- Train and evaluate the network.
- Discuss the impact of architecture choices on performance.

14. Implement and compare the performance of a neural network using different activation
functions (Sigmoid, ReLU, Tanh) on the MNIST dataset. Analyze how each activation
function affects the training process and classification accuracy.
Tasks:
- Load and preprocess the MNIST dataset.
- Implement neural networks using Sigmoid, ReLU, and Tanh activation functions.
- Train and evaluate each network.
- Compare training times, convergence, and classification accuracy.

15. Implement and perform hyperparameter tuning for a neural network on the Fashion
MNIST dataset. Experiment with different learning rates, batch sizes, and epochs, and
discuss the impact on model performance.
Tasks:
- Load and preprocess the Fashion MNIST dataset.
- Experiment with different hyperparameters (learning rate, batch size, epochs).
- Train and evaluate the network.
- Discuss how hyperparameter choices affect model performance.

16. Implement and compare hierarchical (agglomerative) and partitional (K-means) clustering
algorithms on the Mall Customers dataset. Discuss the strengths and weaknesses of each
method based on clustering results and evaluation metrics.
Tasks:
- Load and preprocess the Mall Customers dataset.
- Apply both hierarchical (agglomerative) and K-means clustering.
- Compare results using metrics such as inertia, silhouette score, and clustering
visualization.
- Discuss the advantages and disadvantages of each clustering method.

17. Implement and apply K-means clustering to the Digits dataset. Experiment with different
numbers of clusters and evaluate the clustering results using metrics such as inertia and
silhouette score. Analyze how the choice of K affects clustering performance.
Tasks:
- Load and preprocess the Digits dataset.
- Implement K-means clustering with various numbers of clusters.
- Evaluate clustering performance using inertia and silhouette score.
- Analyze the impact of the number of clusters on clustering quality.

18. Implement bootstrapping and cross-validation on the Iris dataset. Compare the model
performance metrics (e.g., accuracy, F1-score) obtained using these resampling methods.
Discuss the advantages and disadvantages of each method.
Tasks:
- Load and preprocess the Iris dataset.
- Implement bootstrapping to generate multiple samples and evaluate the model.
- Implement k-fold cross-validation and evaluate the model.
- Compare the performance metrics and discuss the pros and cons of each
resampling method.

19. Implement bagging and boosting ensemble methods on the Titanic dataset. Compare the
performance of both methods in terms of accuracy, precision, recall, and F1-score.
Discuss how each method improves model performance and their respective strengths and
weaknesses.
Tasks:
- Load and preprocess the Titanic dataset.
- Implement bagging using a base classifier (e.g., decision tree) and evaluate
performance.
- Implement boosting using a boosting algorithm (e.g., AdaBoost) and evaluate
performance.
- Compare performance metrics and discuss the strengths and weaknesses of each
method.

20. Investigate the bias-variance tradeoff using polynomial regression on the Boston Housing
dataset. Plot the training and validation errors for various polynomial degrees and discuss
the tradeoff between bias and variance.
Tasks:
- Load and preprocess the Boston Housing dataset.
- Implement polynomial regression with varying degrees.
- Plot training and validation errors for each degree.
- Discuss the bias-variance tradeoff and its impact on model performance.



