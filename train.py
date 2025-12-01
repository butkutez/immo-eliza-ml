def handling_csv():
    import pandas as pd

    df = pd.read_csv("./data/clean_data/final_cleaned_data.csv")
    return df

# df = handling_csv()

def train_test_split():
    from sklearn.model_selection import train_test_split
    df = handling_csv()

    # Columns to drop
    columns_to_drop = ["price (€)", "property_ID", "locality_name", "postal_code"]

    X = df.drop(columns=columns_to_drop)
    y = df['price (€)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = train_test_split(df)

def log_transform_target(y_train, y_test):
    import numpy as np
    """
    Apply log1p transform to target and return transformed versions.
    Also returns the inverse_transform function needed after prediction.
    """
    
    # transform (log1p handles zero safely)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # define inverse transform
    def inverse_log_transform(y_pred_log):
        return np.expm1(y_pred_log)

    return y_train_log, y_test_log, inverse_log_transform

def impute_numeric_columns(X_train, X_test):
    from sklearn.impute import SimpleImputer
    """
    Imputes numerical columns in X_train and X_test using the mean of X_train.
    Returns the updated X_train, X_test, and the list of numerical columns.
    """

    # Select numerical columns automatically
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    # Create imputer (mean strategy)
    imputer = SimpleImputer(strategy="mean")

    # Fit on training data
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])

    # Transform test data using SAME means
    X_test[num_cols] = imputer.transform(X_test[num_cols])

    return X_train, X_test, num_cols

# X_train, X_test, num_cols = impute_numeric_columns(X_train, X_test)

def impute_categorical_state(X_train, X_test, column="state_of_building"):
    """
    Fills missing values in the state_of_building column using the 'unknown' category.
    """
    X_train[column] = X_train[column].fillna("unknown")
    X_test[column] = X_test[column].fillna("unknown")
    
    return X_train, X_test

# X_train, X_test = impute_categorical_state(X_train, X_test, column="state_of_building")


def encoding_ohe(X_train, X_test):
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_cols = ["province", "type", "subtype", "state_of_building"]
    # fit only on train to prevent data leakage
    type_train = one_hot_encoder.fit_transform(X_train[ohe_cols])
    type_test = one_hot_encoder.transform(X_test[ohe_cols])

    feature_names = one_hot_encoder.get_feature_names_out(ohe_cols)

    # convert the encoded arrays back into DataFrames
    type_train_df = pd.DataFrame(type_train, columns=feature_names, index=X_train.index)
    type_test_df = pd.DataFrame(type_test, columns=feature_names, index=X_test.index)

    X_train_final = pd.concat([X_train.drop(columns=ohe_cols), type_train_df], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=ohe_cols), type_test_df], axis=1)

    return X_train_final, X_test_final

# X_train_final, X_test_final = encoding_ohe(X_train, X_test)


#     return type_train_df, type_test_df

import pandas as pd
# Step 1: Load CSV
csv = handling_csv()

df = pd.read_csv("./data/clean_data/final_cleaned_data.csv")

# Step 2: Prepare X and y
# X_train, X_test, y_train, y_test = handling_csv()
X_train, X_test, y_train, y_test = train_test_split()

y_train_log, y_test_log, inverse = log_transform_target(y_train, y_test)

# Step 3: Impute numerical values
X_train_imputed_num, X_test_imputed_num, num_cols = impute_numeric_columns(X_train, X_test)

# Step 4: Impute categorical missing values
X_train, X_test = impute_categorical_state(X_train_imputed_num, X_test_imputed_num)

# Step 5: LabelEncoding for "type", "subtype" and "province" columns

X_train_final, X_test_final = encoding_ohe(X_train, X_test)
ohe_cols = ["province", "type", "subtype", "state_of_building"]


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Train XGBoost model
def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=1.0,
        colsample_bytree=0.6,
        random_state=888,
        n_jobs=-1,
        min_child_weight=5
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, 
                   X_train, y_train_log, y_train_real,
                   X_test,  y_test_log,  y_test_real, inverse):
    
    # Predict in log space
    y_pred_train_log = model.predict(X_train)
    y_pred_test_log = model.predict(X_test)
    
    # Convert back to real price scale
    y_pred_train = inverse(y_pred_train_log)
    y_pred_test = inverse(y_pred_test_log)

    metrics = {
        "train": {
            "MAE": mean_absolute_error(y_train_real, y_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_train_real, y_pred_train)),
            "R2": r2_score(y_train_real, y_pred_train)
        },
        "test": {
            "MAE": mean_absolute_error(y_test_real, y_pred_test),
            "RMSE": np.sqrt(mean_squared_error(y_test_real, y_pred_test)),
            "R2": r2_score(y_test_real, y_pred_test)
        }
    }
    return metrics

def run_pipeline():
    # Train XGBoost
    model_xgb = train_xgboost(X_train_final, y_train_log)

    # 7. Evaluate
    results_xgb = evaluate_model(
        model_xgb,
        X_train_final, y_train_log, y_train,
        X_test_final, y_test_log, y_test,
        inverse
    )

    results_xgb

    return model_xgb, results_xgb

if __name__ == "__main__":
    # If this script is run directly, execute the pipeline
    _, _ = run_pipeline()