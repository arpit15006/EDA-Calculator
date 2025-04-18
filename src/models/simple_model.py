import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import joblib
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_application_data

def train_simple_model():
    """
    Train a simple model using only the basic features available in the input form
    
    Returns:
    --------
    tuple
        (model, feature_names)
    """
    # Load data
    app_df = load_application_data()
    
    # Select only the basic features that we'll use for prediction
    basic_features = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'CNT_FAM_MEMBERS'
    ]
    
    # Add derived features that we calculate in the input form
    app_df['CREDIT_TO_INCOME_RATIO'] = app_df['AMT_CREDIT'] / app_df['AMT_INCOME_TOTAL']
    app_df['ANNUITY_TO_INCOME_RATIO'] = app_df['AMT_ANNUITY'] / app_df['AMT_INCOME_TOTAL']
    app_df['INCOME_PER_PERSON'] = app_df['AMT_INCOME_TOTAL'] / app_df['CNT_FAM_MEMBERS']
    app_df['CREDIT_PER_PERSON'] = app_df['AMT_CREDIT'] / app_df['CNT_FAM_MEMBERS']
    app_df['CHILDREN_RATIO'] = app_df['CNT_CHILDREN'] / app_df['CNT_FAM_MEMBERS']
    app_df['IS_WORKING'] = app_df['NAME_INCOME_TYPE'].isin(['Working', 'Commercial associate', 'State servant']).astype(int)
    app_df['IS_PENSIONER'] = (app_df['NAME_INCOME_TYPE'] == 'Pensioner').astype(int)
    app_df['IS_HIGHER_EDUCATION'] = (app_df['NAME_EDUCATION_TYPE'] == 'Higher education').astype(int)
    app_df['IS_MARRIED'] = app_df['NAME_FAMILY_STATUS'].isin(['Married', 'Civil marriage']).astype(int)
    app_df['IS_HOUSE_APARTMENT'] = (app_df['NAME_HOUSING_TYPE'] == 'House / apartment').astype(int)
    app_df['AGE_YEARS'] = abs(app_df['DAYS_BIRTH']) / 365.25
    app_df['EMPLOYMENT_YEARS'] = abs(app_df['DAYS_EMPLOYED']) / 365.25
    
    # Add the derived features to the list
    derived_features = [
        'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO', 'INCOME_PER_PERSON',
        'CREDIT_PER_PERSON', 'CHILDREN_RATIO', 'IS_WORKING', 'IS_PENSIONER',
        'IS_HIGHER_EDUCATION', 'IS_MARRIED', 'IS_HOUSE_APARTMENT',
        'AGE_YEARS', 'EMPLOYMENT_YEARS'
    ]
    
    all_features = basic_features + derived_features
    
    # Clean the data
    app_df = app_df.replace([np.inf, -np.inf], np.nan)
    
    # Split features and target
    X = app_df[all_features]
    y = app_df['TARGET']
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    # Create the model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])
    
    # Fit the model
    print("Training simple model...")
    model.fit(X, y)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/simple_model.pkl')
    
    # Save the feature list
    feature_list = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': all_features
    }
    joblib.dump(feature_list, 'models/feature_list.pkl')
    
    print("Simple model trained and saved.")
    
    return model, feature_list

def load_simple_model():
    """
    Load the simple model and feature list
    
    Returns:
    --------
    tuple
        (model, feature_list)
    """
    if os.path.exists('models/simple_model.pkl') and os.path.exists('models/feature_list.pkl'):
        model = joblib.load('models/simple_model.pkl')
        feature_list = joblib.load('models/feature_list.pkl')
        return model, feature_list
    else:
        return train_simple_model()

if __name__ == "__main__":
    train_simple_model()
