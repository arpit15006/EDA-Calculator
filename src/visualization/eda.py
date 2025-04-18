import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_application_data, load_previous_application_data, load_column_descriptions

def plot_target_distribution(df):
    """
    Plot the distribution of the target variable

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target variable
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='TARGET', data=df)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Target (0: No Default, 1: Default)')
    plt.ylabel('Count')
    plt.savefig('target_distribution.png')
    plt.close()

    # Calculate percentages
    target_counts = df['TARGET'].value_counts(normalize=True) * 100

    # Create a pie chart
    fig = px.pie(
        values=target_counts.values,
        names=target_counts.index.map({0: 'No Default', 1: 'Default'}),
        title='Target Distribution (%)',
        color_discrete_sequence=['#3498db', '#e74c3c'],
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label')

    return fig

def plot_numeric_feature_distributions(df, features, target_col='TARGET'):
    """
    Plot distributions of numeric features by target

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the features
    features : list
        List of numeric features to plot
    target_col : str
        Name of the target column
    """
    figs = []

    for feature in features:
        # Create a figure for the distribution
        fig1 = px.histogram(
            df,
            x=feature,
            title=f'Distribution of {feature}',
            color_discrete_sequence=['#3498db']
        )

        # Create a figure for the boxplot by target
        fig2 = px.box(
            df,
            x=target_col,
            y=feature,
            title=f'{feature} by Target',
            color=target_col,
            color_discrete_sequence=['#3498db', '#e74c3c']
        )

        # Combine the figures
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f'{feature} Distribution', f'{feature} by Target'])

        # Add traces from fig1
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)

        # Add traces from fig2
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)

        fig.update_layout(
            title=f'Distribution of {feature}',
            height=400,
            width=900
        )

        figs.append(fig)

    return figs

def plot_categorical_feature_distributions(df, features, target_col='TARGET'):
    """
    Plot distributions of categorical features by target

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the features
    features : list
        List of categorical features to plot
    target_col : str
        Name of the target column
    """
    figs = []

    for feature in features:
        # Calculate counts and percentages
        counts = df.groupby([feature, target_col]).size().reset_index(name='count')
        total_counts = df.groupby(feature).size().reset_index(name='total')
        counts = counts.merge(total_counts, on=feature)
        counts['percentage'] = counts['count'] / counts['total'] * 100

        # Create a grouped bar chart
        fig = px.bar(
            counts,
            x=feature,
            y='percentage',
            color=target_col,
            barmode='group',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            title=f'{feature} by Target (%)',
            labels={'percentage': 'Percentage (%)'}
        )

        fig.update_layout(height=500, width=900)
        figs.append(fig)

    return figs

def plot_correlation_matrix(df, features):
    """
    Plot correlation matrix for numeric features

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the features
    features : list
        List of numeric features to include in the correlation matrix
    """
    corr_matrix = df[features].corr()

    # Create a heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix'
    )

    fig.update_layout(height=800, width=800)
    return fig

def plot_age_distribution(df):
    """
    Plot age distribution by target

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the DAYS_BIRTH column
    """
    # Convert DAYS_BIRTH to age in years
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25

    fig = px.histogram(
        df,
        x='AGE_YEARS',
        color='TARGET',
        marginal='box',
        nbins=50,
        color_discrete_sequence=['#3498db', '#e74c3c'],
        title='Age Distribution by Target',
        labels={'AGE_YEARS': 'Age (Years)', 'TARGET': 'Default Status'}
    )

    fig.update_layout(height=500, width=900)
    return fig

def plot_income_vs_credit(df):
    """
    Plot income vs credit amount by target

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing income and credit amount columns
    """
    fig = px.scatter(
        df,
        x='AMT_INCOME_TOTAL',
        y='AMT_CREDIT',
        color='TARGET',
        opacity=0.6,
        color_discrete_sequence=['#3498db', '#e74c3c'],
        title='Income vs Credit Amount by Target',
        labels={
            'AMT_INCOME_TOTAL': 'Income Total',
            'AMT_CREDIT': 'Credit Amount',
            'TARGET': 'Default Status'
        }
    )

    fig.update_layout(height=600, width=900)
    return fig

def run_eda():
    """
    Run exploratory data analysis and return figures

    Returns:
    --------
    dict
        Dictionary containing plotly figures
    """
    # Load data
    app_df = load_application_data()
    prev_df = load_previous_application_data()

    # Basic data cleaning
    # Convert days to positive values and more intuitive features
    app_df['DAYS_BIRTH'] = abs(app_df['DAYS_BIRTH'])
    app_df['DAYS_EMPLOYED'] = abs(app_df['DAYS_EMPLOYED'])

    # Replace anomalous values
    app_df.loc[app_df['DAYS_EMPLOYED'] > 365*100, 'DAYS_EMPLOYED'] = np.nan

    # Create age and employment length features
    app_df['AGE_YEARS'] = app_df['DAYS_BIRTH'] / 365.25
    app_df['EMPLOYMENT_YEARS'] = app_df['DAYS_EMPLOYED'] / 365.25

    # Create credit to income ratio
    app_df['CREDIT_TO_INCOME_RATIO'] = app_df['AMT_CREDIT'] / app_df['AMT_INCOME_TOTAL']
    app_df['ANNUITY_TO_INCOME_RATIO'] = app_df['AMT_ANNUITY'] / app_df['AMT_INCOME_TOTAL']

    # Select important numeric features for analysis
    numeric_features = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO'
    ]

    # Select important categorical features for analysis
    categorical_features = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'
    ]

    # Generate plots
    figures = {}

    # Target distribution
    figures['target_distribution'] = plot_target_distribution(app_df)

    # Numeric feature distributions
    figures['numeric_distributions'] = plot_numeric_feature_distributions(
        app_df,
        ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO']
    )

    # Categorical feature distributions
    figures['categorical_distributions'] = plot_categorical_feature_distributions(
        app_df,
        ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE']
    )

    # Correlation matrix
    figures['correlation_matrix'] = plot_correlation_matrix(
        app_df,
        ['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH',
         'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO']
    )

    # Age distribution
    figures['age_distribution'] = plot_age_distribution(app_df)

    # Income vs Credit
    figures['income_vs_credit'] = plot_income_vs_credit(app_df)

    return figures, app_df

if __name__ == "__main__":
    run_eda()
