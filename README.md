# supervised-machine-learning-challenge# Supervised Machine Learning

## Background
### Lending services companies allow individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market.


### You will be using this data to create machine learning models to classify the risk level of given loans. Specifically, you will be comparing the Logistic Regression model and Random Forest Classifier.

## Installations needed
### numpy, pandas, sklearn, train_test_split, balanced_accuracy_score, confusion_matrix, RandomForestClassifier, LogisticRegression

## Aim
### The aim of this project is to create both a Logistic Regression and Random Forrest classifier and compare the accuracy of both models. In order to do this, I first split the data into training and testing data. I then created and fit both models with the training and testing data. I chose to use the default hyperparameters. I then displayed the results in a confusion matrix.

## Example of code used
### The following code is an example of how I classified the training, testing and splitting of the data.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


![Logistic Regression Confusion Matrix](https://github.com/nickjaycarr88/supervised-machine-learning-challenge/blob/main/logistic_regression_confusion_matrix.png)


![Random Forrest Confusion Matrix](https://github.com/nickjaycarr88/supervised-machine-learning-challenge/blob/main/random_forrest_confusion_matrix.png)
