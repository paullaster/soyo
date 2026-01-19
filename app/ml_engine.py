import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_absolute_error

# 1. Load Data
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# 2. Preprocess
def preprocess_data(df):
    # Feature Engineering
    # We want to predict risk based on available info AT TIME OF AWARD
    
    # Target 1: Risk Class
    # High Risk = Delayed > 30 days OR Overrun > 10% OR Terminated
    # Medium Risk = Delayed > 7 days OR Overrun > 0%
    # Low Risk = Everything else
    
    conditions = [
        (df['contract_status'] == 'Terminated') | (df['days_delayed'] > 30) | (df['cost_overrun_percentage'] > 10),
        (df['days_delayed'] > 7) | (df['cost_overrun_percentage'] > 0)
    ]
    choices = ['High', 'Medium']
    df['risk_level'] = np.select(conditions, choices, default='Low')
    
    # Features for training
    feature_cols = ['tender_budget_kes', 'credit_score', 'company_size', 'supplier_age_at_award_days', 'category']
    
    X = df[feature_cols]
    y_class = df['risk_level']
    y_reg = df['days_delayed']
    
    return X, y_class, y_reg

# 3. Build Pipelines
def build_pipelines():
    numeric_features = ['tender_budget_kes', 'credit_score', 'supplier_age_at_award_days']
    categorical_features = ['company_size', 'category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    return clf_pipeline, reg_pipeline

# 4. Train & Save
def train_and_save():
    print("Loading data...")
    df = load_data('data/procurement_master_dataset.csv')
    
    print("Preprocessing...")
    X, y_class, y_reg = preprocess_data(df)
    
    # Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    clf_model, reg_model = build_pipelines()
    
    print("Training Risk Classifier...")
    clf_model.fit(X_train, y_class_train)
    print("Classifier Performance:")
    print(classification_report(y_class_test, clf_model.predict(X_test)))
    
    print("Training Delay Regressor...")
    reg_model.fit(X_train, y_reg_train)
    mae = mean_absolute_error(y_reg_test, reg_model.predict(X_test))
    print(f"Regressor MAE: {mae:.2f} days")
    
    print("Saving models...")
    joblib.dump(clf_model, 'models/risk_classifier.pkl')
    joblib.dump(reg_model, 'models/delay_regressor.pkl')
    print("Models saved to models/")

if __name__ == "__main__":
    train_and_save()
