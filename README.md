# Immo-eliza-ml

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Property_Price_Predictor](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*nVnCGQ2I6mTlx0whCxREFA.jpeg)](https://medium.com/@bravinwasike18/dive-into-xgboost-and-scikit-learnmachine-learning-with-xgboost-and-scikit-learn-17e2cf54f3a3)  
*Image source: [Medium](https://medium.com/@bravinwasike18/dive-into-xgboost-and-scikit-learnmachine-learning-with-xgboost-and-scikit-learn-17e2cf54f3a3)*

## Description

The real estate company Immo Eliza is looking to predict property prices in Belgium. This project develops a machine learning pipeline to preprocess a scraped real estate dataset and train multiple models to achieve the best price prediction performance.

This repository contains the complete ML solution, from data preprocessing techniques like imputation and one-hot encoding, through training and evaluation of baseline and advanced XGBoost model, to a final, deployable prediction script.

## The Mission

To build a performant machine learning model to accurately predict the sale price of real estate properties in Belgium, utilizing a robust and reusable data processing and modeling pipeline.

## Technical Implementation
**Model Implemented**

XGBoost - Advanced gradient boosting for improved performance

**Data Pipeline**  
- Data Cleaning: Handling duplicates, missing values, and irrelevant columns
- Preprocessing: Imputation, one-hot encoding
- Model Training: XGBoost - Advanced gradient boosting for improved performance

## Performance Metrics
Models were evaluated using:
- R-squared (R²)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Cross-validation scores

## Repo Structure

```
IMMO-ELIZA-ML
├── data
│     ├── clean_data
│     │      └── final_cleaned_data.csv
│     └── raw_data
│            └── cleaned_data_int.csv
├── 3_XGBoost_notebook.ipynb
├── adding_a_column_to_csv.ipynb
├── main.py
├── model.pkl
├── pipeline.py
├── README.md
├── requirements.txt
└── train.py
```

## Installation

1. **Clone the project:**

```
git clone https://github.com/butkutez/immo-eliza-ml.git
```
2. **Navigate into the project folder**

```
cd immo-eliza-ml
```

3. **Install dependencies:**

```
pip install -r requirements.txt
```

## Running the Project
````
# Run the model
python pipeline.py
````

## Summary
In this project data was sucessfully preprocessed (imputation/encoding) within the pipeline, then trained with optimized XGBoost hyperparameters.

The model is evaluated using inverse-transformed predictions and the final pipeline is saved as model.pkl.

Future Improvements:  
- Correlation Analysis (Specific Feature Selection)
- Test Different Models (Benchmarking)        

## **Timeline**
This project was completed over 4 days.

## **Personal Situation**
This project was completed as part of the AI & Data Science Bootcamp at BeCode.org.

**Connect** with me on [LinkedIn](https://www.linkedin.com/in/zivile-butkute/).




