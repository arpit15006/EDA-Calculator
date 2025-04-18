import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib
import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessor import prepare_data_for_modeling

def train_logistic_regression(X_train, y_train, preprocessor, cv=5):
    """
    Train a logistic regression model

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    preprocessor : ColumnTransformer
        Feature preprocessor
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    tuple
        (model, cv_scores)
    """
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc'
    )

    # Fit the model on the entire training set
    model.fit(X_train, y_train)

    return model, cv_scores

def train_random_forest(X_train, y_train, preprocessor, cv=5):
    """
    Train a random forest model

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    preprocessor : ColumnTransformer
        Feature preprocessor
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    tuple
        (model, cv_scores)
    """
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ))
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc'
    )

    # Fit the model on the entire training set
    model.fit(X_train, y_train)

    return model, cv_scores

def train_gradient_boosting(X_train, y_train, preprocessor, cv=5):
    """
    Train a gradient boosting model

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    preprocessor : ColumnTransformer
        Feature preprocessor
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    tuple
        (model, cv_scores)
    """
    # Create pipeline with preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        ))
    ])

    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc'
    )

    # Fit the model on the entire training set
    model.fit(X_train, y_train)

    return model, cv_scores

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on the test set

    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target

    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    metrics['roc_curve'] = (fpr, tpr)

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    metrics['pr_curve'] = (precision, recall)
    metrics['pr_auc'] = auc(recall, precision)

    return metrics

def plot_roc_curves(models_metrics, model_names):
    """
    Plot ROC curves for multiple models

    Parameters:
    -----------
    models_metrics : list
        List of model metrics dictionaries
    model_names : list
        List of model names

    Returns:
    --------
    plt.Figure
        Figure object
    """
    plt.figure(figsize=(10, 8))

    for i, metrics in enumerate(models_metrics):
        fpr, tpr = metrics['roc_curve']
        roc_auc = metrics['roc_auc']
        plt.plot(fpr, tpr, lw=2, label=f'{model_names[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    return plt.gcf()

def plot_pr_curves(models_metrics, model_names):
    """
    Plot Precision-Recall curves for multiple models

    Parameters:
    -----------
    models_metrics : list
        List of model metrics dictionaries
    model_names : list
        List of model names

    Returns:
    --------
    plt.Figure
        Figure object
    """
    plt.figure(figsize=(10, 8))

    for i, metrics in enumerate(models_metrics):
        precision, recall = metrics['pr_curve']
        pr_auc = metrics['pr_auc']
        plt.plot(recall, precision, lw=2, label=f'{model_names[i]} (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")

    return plt.gcf()

def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model

    Parameters:
    -----------
    model : Pipeline
        Trained model pipeline
    feature_names : list
        List of feature names

    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importances
    """
    # Get the classifier from the pipeline
    classifier = model.named_steps['classifier']

    # Get the preprocessor from the pipeline
    preprocessor = model.named_steps['preprocessor']

    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names_out = preprocessor.get_feature_names_out()
    else:
        # For older scikit-learn versions
        feature_names_out = feature_names

    # Get feature importances based on the classifier type
    if hasattr(classifier, 'feature_importances_'):
        # For tree-based models
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # For linear models
        importances = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame({'feature': feature_names_out, 'importance': np.zeros(len(feature_names_out))})

    # Create a DataFrame with feature importances
    feature_importance = pd.DataFrame({
        'feature': feature_names_out,
        'importance': importances
    })

    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    return feature_importance

def train_and_evaluate_models(use_pretrained=True):
    """
    Train and evaluate multiple models

    Parameters:
    -----------
    use_pretrained : bool
        Whether to use pretrained models if available

    Returns:
    --------
    tuple
        (models, metrics, feature_importances)
    """
    # Check if pretrained models exist
    if use_pretrained and os.path.exists('models/logistic_regression.pkl') and \
       os.path.exists('models/random_forest.pkl') and \
       os.path.exists('models/gradient_boosting.pkl'):
        print("Loading pretrained models...")
        lr_model = joblib.load('models/logistic_regression.pkl')
        rf_model = joblib.load('models/random_forest.pkl')
        gb_model = joblib.load('models/gradient_boosting.pkl')

        # Prepare data for evaluation only
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling(sample_size=5000)

        # Evaluate models
        print("Evaluating models...")
        lr_metrics = evaluate_model(lr_model, X_test, y_test)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        gb_metrics = evaluate_model(gb_model, X_test, y_test)

        # Get feature importances
        lr_importance = get_feature_importance(lr_model, feature_names)
        rf_importance = get_feature_importance(rf_model, feature_names)
        gb_importance = get_feature_importance(gb_model, feature_names)

        # Create dummy CV scores (since we're not doing CV with pretrained models)
        lr_cv_scores = np.array([lr_metrics['roc_auc']])
        rf_cv_scores = np.array([rf_metrics['roc_auc']])
        gb_cv_scores = np.array([gb_metrics['roc_auc']])
    else:
        # Prepare data for training
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data_for_modeling(sample_size=10000)

        # Train models
        print("Training Logistic Regression...")
        lr_model, lr_cv_scores = train_logistic_regression(X_train, y_train, preprocessor)

        print("Training Random Forest...")
        rf_model, rf_cv_scores = train_random_forest(X_train, y_train, preprocessor)

        print("Training Gradient Boosting...")
        gb_model, gb_cv_scores = train_gradient_boosting(X_train, y_train, preprocessor)

        # Evaluate models
        print("Evaluating models...")
        lr_metrics = evaluate_model(lr_model, X_test, y_test)
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        gb_metrics = evaluate_model(gb_model, X_test, y_test)

        # Get feature importances
        lr_importance = get_feature_importance(lr_model, feature_names)
        rf_importance = get_feature_importance(rf_model, feature_names)
        gb_importance = get_feature_importance(gb_model, feature_names)

    # Combine results
    models = {
        'logistic_regression': lr_model,
        'random_forest': rf_model,
        'gradient_boosting': gb_model
    }

    metrics = {
        'logistic_regression': {
            'cv_scores': lr_cv_scores,
            'test_metrics': lr_metrics
        },
        'random_forest': {
            'cv_scores': rf_cv_scores,
            'test_metrics': rf_metrics
        },
        'gradient_boosting': {
            'cv_scores': gb_cv_scores,
            'test_metrics': gb_metrics
        }
    }

    feature_importances = {
        'logistic_regression': lr_importance,
        'random_forest': rf_importance,
        'gradient_boosting': gb_importance
    }

    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    joblib.dump(gb_model, 'models/gradient_boosting.pkl')

    return models, metrics, feature_importances

if __name__ == "__main__":
    start_time = time.time()
    models, metrics, feature_importances = train_and_evaluate_models()
    end_time = time.time()

    print(f"\nTraining and evaluation completed in {end_time - start_time:.2f} seconds.")

    # Print cross-validation results
    print("\nCross-Validation Results (ROC AUC):")
    for model_name, model_metrics in metrics.items():
        cv_scores = model_metrics['cv_scores']
        print(f"{model_name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Print test metrics
    print("\nTest Set Results (ROC AUC):")
    for model_name, model_metrics in metrics.items():
        test_metrics = model_metrics['test_metrics']
        print(f"{model_name}: {test_metrics['roc_auc']:.4f}")

    # Print top features
    print("\nTop 10 Features (Gradient Boosting):")
    print(feature_importances['gradient_boosting'].head(10))
