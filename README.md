# Regression

- Repository: `immo-eliza-ml`
- Type: `Consolidation`
- Duration: `4 days`
- Deadline: `27/11/2024 5:00 PM`
- Show and tell: `01/12/2024 9:30 - 10:30 AM`
- Team: solo

## Learning Objectives

- Be able to preprocess data for machine learning.
- Be able to apply a linear regression in a real-life context.
- Be able to explore machine learning models for regression.
- Be able to evaluate the performance of a model
- (Optional) Be able to apply hyperparameter tuning and cross-validation.

## The Mission

The real estate company Immo Eliza asked you to create a machine learning model to predict prices of real estate properties in Belgium.

After the **scraping**, **cleaning** and **analyzing**, you are ready to preprocess the data and finally build a performant machine learning model!

## Steps

You still need to further prepare the dataset for machine learning. Think about:
- Handling NaNs (hint: **imputation**)
- Converting categorical data into numeric features (hint: **one-hot encoding**)
- Rescaling numeric features (hint: **standardization**)

**Keep track of your preprocessing steps. You will need to apply the same steps to the new dataset later on when generating predictions.** This is crucial! Think _reusable pipeline_!

Additionally, you can also consider (but feel free to skip this and go straight to a first model iteration):
- Preselecting only the features that have at least some correlation with your target variable (hint: **univariate regression** or **correlation matrix**)
- Removing features that have too strong correlation with one another (hint: **correlation matrix**)

Once done, split your dataset for training and testing.

### Model training

The dataset is ready. Let's select a model.

Start with **linear regression**, this will give you a good baseline performance. You will compare this baseline with two non-linear models: one between ([Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor), [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)), and [XGBoost](https://xgboost.readthedocs.io/en/stable/). At the end, you should have 3 models tested. 

Train your model(s) and save it! (hint: search for libraries like pickle and joblib)

### Model evaluation & iteration

Evaluate your model. What do the performance metrics say?

### Improve

Don't stop with your first model. Iterate! Try to improve your model by testing different models, or by changing the preprocessing.

## Technical Pipeline

You should create a model on the training data, and test it on your proper test data obtained after splitting the dataset. Even in your notebook, apply good practices such as working with functions, commenting, etc. For example, you could create the functions for the main steps of the data pipeline:
- clean data
    - Handling duplicates, missing values, droping columns or rows
- preprocess data
    - Imputing missing values, encoding, rescaling
- train model
    - Spliting data, fitting model
- predict 
- evaluate model
    - Compute appropiate metrics R-squared, MSE, MAE, etc.
    - Is there overfitting? Yes/No, how do you know?


You could also create a `predict.py` script that uses the save model created by your `train.py`, load it, and use it to predict the price of a new house. 

**NOTE:** Start simple, build big! For example, as a first attempt you could decide to drop all missing values, or only work with numerical features, or focus on a single model and as you build your code, add new features!

## Deliverables

1. Publish your code and model(s) on a GitHub repository named `immo-eliza-ml`
    - Don't forget to do the virtual environment, .gitignore, ... dance
2. README is non-negotiable!
   - It should include the usual sections plus a summary of your results and approach. 

3. Show and tell! We will pick 2-3 random people to present their notebook/repo.

## Evaluation criteria

| Criteria       | Indicator                                                    | Yes/No |
| -------------- | ------------------------------------------------------------ | ------ |
| 1. Is good     | Your repository is complete                                  | [ ]    |
|                | Your code is clean                                           | [ ]    |
|                | Your README is clean and complete                                    | [ ]    |
|                | Your `predict.py` with new dummy data  | [ ]    |
## Quotes

_"Artificial intelligence, deep learning, machine learning — whatever you're doing, if you don't understand it — learn it. Because otherwise you're going to be a dinosaur within 3 years." - Mark Cuban_

![You've got this!](https://media.giphy.com/media/5wWf7GMbT1ZUGTDdTqM/giphy.gif)