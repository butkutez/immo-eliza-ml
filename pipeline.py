from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from train import handling_csv
from train import log_transform_target
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from train import train_test_split_fun
from train import train_xgboost
from train import evaluate_model
from train import run_pipeline
from xgboost import XGBRegressor

import pandas as pd
# Step 1: Load CSV
csv = handling_csv()
df = pd.read_csv("./data/clean_data/final_cleaned_data.csv")

# Step 2: Prepare X and y
# X_train, X_test, y_train, y_test = handling_csv()
X_train, X_test, y_train, y_test = train_test_split_fun()
y_train_log, y_test_log, inverse = log_transform_target(y_train, y_test)

# Select numerical columns
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean', fill_value=-1, add_indicator=True))])

# Select categorical columns
categorical_features = ["province", "type", "subtype", "state_of_building"]
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='unknown')), ('encoder', OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=1.0,
        colsample_bytree=0.6,
        random_state=888,
        n_jobs=-1,
        min_child_weight=5)
    )]
)


clf.fit(X_train, y_train_log)
eval_model = evaluate_model(clf,X_train, y_train,
                   X_test,  y_test, inverse)

joblib.dump(eval_model, "model.pkl")