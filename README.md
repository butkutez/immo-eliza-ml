# Immo-eliza-ml

- Repository: `immo-eliza-ml`
- Type: `Consolidation`
- Duration: `4 days`
- Deadline: `27/11/2024 5:00 PM`
- Show and tell: `01/12/2024 9:30 - 10:30 AM`
- Team: solo

## Description

The real estate company Immo Eliza is looking to predict property prices in Belgium. This project develops a machine learning pipeline to preprocess a scraped real estate dataset and train multiple models to achieve the best price prediction performance.

This repository contains the complete ML solution, from data preprocessing techniques like imputation and one-hot encoding, through training and evaluation of baseline and advanced models, to a final, deployable prediction script.

## The Mission

To build a performant machine learning model to accurately predict the sale price of real estate properties in Belgium, utilizing a robust and reusable data processing and modeling pipeline.

## Technical Implementation
### Model Implemented

1) XGBoost - Advanced gradient boosting for improved performance

### Data Pipeline
* Data Cleaning: Handling duplicates, missing values, and irrelevant columns
* Preprocessing: Imputation, one-hot encoding
* Model Training: XGBoost - Advanced gradient boosting for improved performance

## Performance Metrics
Models were evaluated using:

* R-squared (R²)
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Cross-validation scores

## Repo Structure

```
.
├── IMMO-ELIZA-ML/
│   │
│   └── data
│         ├── clean_data
│         │      └── final_cleaned_data.csv
│         └── raw_data
│                └── cleaned_data_int.csv
├── 3_XGBoost_notebook.ipynb
├── adding_a_column_to_csv.ipynb
├── main.py
├── train.py
├── README.md
└── requirements.txt
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

3. **Set up the virtual environment (recommended):**

```
python -m venv venv
venv\Scripts\activate
```

4. **Install dependencies:**

```
pip install -r requirements.txt
```

## Running the Project
````
# Train models
python train.py

# Run the model
python main.py
````

## Results Summary
The project implemented XGBoost prediction models.

Future Improvements
* Feature Selection: Correlation analysis and multicollinearity handling
* Model Training: Three different algorithms with proper train-test split
* Evaluation: Comprehensive metrics and overfitting analysis
* Hyperparameter optimization with GridSearchCV
* Explore more regularization techniques
* Feature engineering for additional predictive power
* Integration in pipeline. 
                          
<br>               

*Project completed as part of machine learning consolidation during the BeCode Data Science & AI course - Duration: 4 days*



