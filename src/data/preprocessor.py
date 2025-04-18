import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_application_data, load_previous_application_data, merge_application_with_previous

def preprocess_application_data(df):
    """
    Preprocess the application data

    Parameters:
    -----------
    df : pd.DataFrame
        Application data DataFrame

    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()

    # Handle days features (convert to positive and more intuitive)
    days_cols = [col for col in df_processed.columns if col.startswith('DAYS_')]
    for col in days_cols:
        df_processed[col] = df_processed[col].abs()

    # Replace anomalous values in DAYS_EMPLOYED
    df_processed.loc[df_processed['DAYS_EMPLOYED'] > 365*100, 'DAYS_EMPLOYED'] = np.nan

    # Create age and employment length features
    df_processed['AGE_YEARS'] = df_processed['DAYS_BIRTH'] / 365.25
    df_processed['EMPLOYMENT_YEARS'] = df_processed['DAYS_EMPLOYED'] / 365.25

    # Create credit to income ratio
    df_processed['CREDIT_TO_INCOME_RATIO'] = df_processed['AMT_CREDIT'] / df_processed['AMT_INCOME_TOTAL']
    df_processed['ANNUITY_TO_INCOME_RATIO'] = df_processed['AMT_ANNUITY'] / df_processed['AMT_INCOME_TOTAL']

    # Create income per family member
    df_processed['INCOME_PER_PERSON'] = df_processed['AMT_INCOME_TOTAL'] / (df_processed['CNT_FAM_MEMBERS'] + 1)

    # Create credit per person
    df_processed['CREDIT_PER_PERSON'] = df_processed['AMT_CREDIT'] / (df_processed['CNT_FAM_MEMBERS'] + 1)

    # Create children ratio
    df_processed['CHILDREN_RATIO'] = df_processed['CNT_CHILDREN'] / (df_processed['CNT_FAM_MEMBERS'] + 1)

    # Create income type dummies
    df_processed['IS_WORKING'] = df_processed['NAME_INCOME_TYPE'].isin(['Working', 'Commercial associate', 'State servant']).astype(int)
    df_processed['IS_PENSIONER'] = (df_processed['NAME_INCOME_TYPE'] == 'Pensioner').astype(int)

    # Create education type dummies
    df_processed['IS_HIGHER_EDUCATION'] = (df_processed['NAME_EDUCATION_TYPE'] == 'Higher education').astype(int)

    # Create family status dummies
    df_processed['IS_MARRIED'] = df_processed['NAME_FAMILY_STATUS'].isin(['Married', 'Civil marriage']).astype(int)

    # Create housing type dummies
    df_processed['IS_HOUSE_APARTMENT'] = (df_processed['NAME_HOUSING_TYPE'] == 'House / apartment').astype(int)

    # Create external source mean
    ext_source_cols = [col for col in df_processed.columns if col.startswith('EXT_SOURCE_')]
    df_processed['EXT_SOURCE_MEAN'] = df_processed[ext_source_cols].mean(axis=1)

    return df_processed

def create_feature_pipeline(df):
    """
    Create a feature preprocessing pipeline

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to create pipeline for

    Returns:
    --------
    tuple
        (preprocessor, numeric_features, categorical_features)
    """
    # Identify numeric and categorical features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target and ID from features
    if 'TARGET' in numeric_features:
        numeric_features.remove('TARGET')
    if 'SK_ID_CURR' in numeric_features:
        numeric_features.remove('SK_ID_CURR')
    if 'SK_ID_PREV' in numeric_features:
        numeric_features.remove('SK_ID_PREV')

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

    return preprocessor, numeric_features, categorical_features

def prepare_data_for_modeling(train_size=0.8, random_state=42, sample_size=10000):
    """
    Prepare data for modeling

    Parameters:
    -----------
    train_size : float
        Proportion of data to use for training
    random_state : int
        Random seed for reproducibility
    sample_size : int
        Number of samples to use for training (for faster processing)

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    # Load and merge data
    app_df = load_application_data()
    prev_df = load_previous_application_data()
    merged_df = merge_application_with_previous(app_df, prev_df)

    # Preprocess data
    processed_df = preprocess_application_data(merged_df)

    # Take a smaller sample for faster processing
    if sample_size and sample_size < len(processed_df):
        # Make sure we have a balanced sample
        df_majority = processed_df[processed_df['TARGET'] == 0]
        df_minority = processed_df[processed_df['TARGET'] == 1]

        # Calculate the sample size for each class
        minority_size = min(sample_size // 4, len(df_minority))
        majority_size = min(sample_size - minority_size, len(df_majority))

        # Sample from each class
        df_majority_sampled = df_majority.sample(majority_size, random_state=random_state)
        df_minority_sampled = df_minority.sample(minority_size, random_state=random_state)

        # Combine the samples
        processed_df = pd.concat([df_majority_sampled, df_minority_sampled])

        print(f"Using a sample of {len(processed_df)} records (original size: {len(merged_df)})")

    # Split features and target
    X = processed_df.drop(['TARGET'], axis=1)
    y = processed_df['TARGET']

    # Create preprocessing pipeline
    preprocessor, numeric_features, categorical_features = create_feature_pipeline(X)

    # Get feature names
    feature_names = numeric_features + categorical_features

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    # Test the functions
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Number of features:", len(feature_names))
    print("Target distribution in training set:")
    print(y_train.value_counts(normalize=True) * 100)
