import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("loan.csv")
data = data.dropna(how='any', inplace=False)
data = data.drop(["Loan_ID", "Education", "CoapplicantIncome"], axis=1)

# Features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"].map({"Y": 1, "N": 0})  # ✅ Encode target (Y=1, N=0)

# Encode categorical features
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Property_Area"]
numeric_cols = ["ApplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# XGBoost model
xgb = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", xgb)])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy", verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Best Parameters:", grid_search.best_params_)

# Save model
joblib.dump(best_model, "loan.joblib")
print("✅ Stronger XGBoost Model trained and saved as loan.joblib")
