import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from scipy import stats

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_application_data, load_previous_application_data, load_column_descriptions

def load_and_preprocess_data():
    """
    Load and preprocess the data

    Returns:
    --------
    tuple
        (app_df, prev_df, col_desc, app_df_cleaned, missing_values_summary)
    """
    # Load data
    app_df = load_application_data()
    prev_df = load_previous_application_data()
    col_desc = load_column_descriptions()

    # Check for duplicates
    app_duplicates = app_df.duplicated().sum()
    prev_duplicates = prev_df.duplicated().sum()

    print(f"Application data duplicates: {app_duplicates}")
    print(f"Previous application data duplicates: {prev_duplicates}")

    # Calculate missing values before cleaning
    missing_before = pd.DataFrame({
        'column': app_df.columns,
        'missing_count': app_df.isnull().sum(),
        'missing_percentage': round((app_df.isnull().sum() / len(app_df) * 100), 2)
    }).sort_values('missing_percentage', ascending=False)

    # Create a copy for cleaning
    app_df_cleaned = app_df.copy()

    # Remove columns with >50% missing data
    cols_to_drop = missing_before[missing_before['missing_percentage'] > 50]['column'].tolist()
    app_df_cleaned = app_df_cleaned.drop(columns=cols_to_drop)

    # Identify numerical and categorical columns
    num_cols = app_df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = app_df_cleaned.select_dtypes(include=['object']).columns.tolist()

    # Remove ID and target from numerical columns
    if 'SK_ID_CURR' in num_cols:
        num_cols.remove('SK_ID_CURR')
    if 'TARGET' in num_cols:
        num_cols.remove('TARGET')

    # Impute numerical variables with median
    for col in num_cols:
        app_df_cleaned[col] = app_df_cleaned[col].fillna(app_df_cleaned[col].median())

    # Replace categorical missing values with "Unknown"
    for col in cat_cols:
        app_df_cleaned[col] = app_df_cleaned[col].fillna("Unknown")

    # Calculate missing values after cleaning
    missing_after = pd.DataFrame({
        'column': app_df_cleaned.columns,
        'missing_count': app_df_cleaned.isnull().sum(),
        'missing_percentage': round((app_df_cleaned.isnull().sum() / len(app_df_cleaned) * 100), 2)
    }).sort_values('missing_percentage', ascending=False)

    # Create summary table
    missing_values_summary = pd.DataFrame({
        'column': missing_before['column'],
        'missing_percentage_before': missing_before['missing_percentage'],
        'missing_percentage_after': missing_before['column'].map(
            dict(zip(missing_after['column'], missing_after['missing_percentage']))
        ).fillna(0)
    }).sort_values('missing_percentage_before', ascending=False)

    return app_df, prev_df, col_desc, app_df_cleaned, missing_values_summary

def detect_outliers(df, columns):
    """
    Detect outliers in numerical columns using IQR method

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    columns : list
        List of numerical columns to check for outliers

    Returns:
    --------
    tuple
        (outlier_summary, outlier_figures)
    """
    outlier_summary = pd.DataFrame(columns=['column', 'total_count', 'outlier_count', 'outlier_percentage', 'min', 'q1', 'median', 'q3', 'max'])
    outlier_figures = []

    for col in columns:
        # Calculate IQR
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = round((outlier_count / len(df) * 100), 2)

        # Add to summary
        outlier_summary = pd.concat([outlier_summary, pd.DataFrame({
            'column': [col],
            'total_count': [len(df)],
            'outlier_count': [outlier_count],
            'outlier_percentage': [outlier_percentage],
            'min': [df[col].min()],
            'q1': [q1],
            'median': [df[col].median()],
            'q3': [q3],
            'max': [df[col].max()]
        })], ignore_index=True)

        # Create boxplot
        fig = px.box(
            df,
            y=col,
            title=f'Boxplot of {col} (Outlier %: {outlier_percentage}%)',
            points="outliers"
        )

        # Add annotations for bounds
        fig.add_hline(y=lower_bound, line_dash="dash", line_color="red",
                     annotation_text="Lower bound", annotation_position="bottom right")
        fig.add_hline(y=upper_bound, line_dash="dash", line_color="red",
                     annotation_text="Upper bound", annotation_position="top right")

        outlier_figures.append(fig)

        # Check for extreme values
        if col == 'AMT_INCOME_TOTAL':
            extreme_high = df[df[col] > 1000000]
            print(f"Extreme high income (>$1M): {len(extreme_high)} cases ({(len(extreme_high)/len(df)*100):.2f}%)")

        if col == 'DAYS_EMPLOYED':
            extreme_high = df[df[col] > 365*100]  # More than 100 years
            print(f"Extreme employment duration (>100 years): {len(extreme_high)} cases ({(len(extreme_high)/len(df)*100):.2f}%)")

    return outlier_summary, outlier_figures

def check_data_imbalance(df):
    """
    Check for data imbalance in the target variable

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the target variable

    Returns:
    --------
    tuple
        (imbalance_summary, imbalance_figure)
    """
    # Calculate target distribution
    target_counts = df['TARGET'].value_counts()
    target_percentages = round((df['TARGET'].value_counts(normalize=True) * 100), 2)

    # Create summary
    imbalance_summary = pd.DataFrame({
        'target_value': target_counts.index,
        'count': target_counts.values,
        'percentage': target_percentages.values
    })

    # Calculate imbalance ratio
    imbalance_ratio = target_counts[0] / target_counts[1]

    # Create pie chart
    imbalance_figure = px.pie(
        values=target_counts.values,
        names=['No Default (0)', 'Default (1)'],
        title=f'Target Distribution (Imbalance Ratio: {imbalance_ratio:.2f}:1)',
        color_discrete_sequence=['#3498db', '#e74c3c'],
        hole=0.4
    )

    imbalance_figure.update_traces(textinfo='percent+label')

    return imbalance_summary, imbalance_figure

def univariate_analysis(df, numerical_cols, categorical_cols):
    """
    Perform univariate analysis on numerical and categorical columns

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    numerical_cols : list
        List of numerical columns to analyze
    categorical_cols : list
        List of categorical columns to analyze

    Returns:
    --------
    tuple
        (numerical_figures, categorical_figures)
    """
    numerical_figures = []
    categorical_figures = []

    # Analyze numerical columns
    for col in numerical_cols:
        # Create histogram
        fig = px.histogram(
            df,
            x=col,
            color='TARGET',
            barmode='overlay',
            marginal='box',
            title=f'Distribution of {col} by Target',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )

        numerical_figures.append(fig)

        # Calculate statistics by target
        stats_by_target = df.groupby('TARGET')[col].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
        print(f"\nStatistics for {col} by TARGET:")
        print(stats_by_target)

    # Analyze categorical columns
    for col in categorical_cols:
        # Calculate value counts and percentages
        value_counts = df.groupby([col, 'TARGET']).size().reset_index(name='count')
        total_counts = df.groupby(col).size().reset_index(name='total')
        value_counts = value_counts.merge(total_counts, on=col)
        value_counts['percentage'] = round((value_counts['count'] / value_counts['total'] * 100), 2)

        # Create stacked bar chart
        fig = px.bar(
            value_counts,
            x=col,
            y='count',
            color='TARGET',
            barmode='group',
            title=f'Distribution of {col} by Target',
            color_discrete_sequence=['#3498db', '#e74c3c'],
            text='percentage'
        )

        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        categorical_figures.append(fig)

    return numerical_figures, categorical_figures

def bivariate_analysis(df):
    """
    Perform bivariate analysis and correlation analysis

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data

    Returns:
    --------
    tuple
        (correlation_figures, scatter_figures)
    """
    correlation_figures = []
    scatter_figures = []

    # Select numerical columns for correlation analysis
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Remove ID and target from correlation columns
    if 'SK_ID_CURR' in num_cols:
        num_cols.remove('SK_ID_CURR')
    if 'TARGET' in num_cols:
        num_cols.remove('TARGET')

    # Calculate correlation matrix for all data
    corr_matrix = df[num_cols].corr()

    # Create heatmap for all data
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix (All Data)'
    )

    correlation_figures.append(fig)

    # Calculate correlation matrices for each target segment
    for target_val in [0, 1]:
        segment_df = df[df['TARGET'] == target_val]
        segment_corr = segment_df[num_cols].corr()

        # Create heatmap for segment
        fig = px.imshow(
            segment_corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title=f'Correlation Matrix (TARGET = {target_val})'
        )

        correlation_figures.append(fig)

        # Find top 10 correlated pairs
        corr_pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                corr_pairs.append((num_cols[i], num_cols[j], abs(segment_corr.iloc[i, j])))

        top_corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:10]

        print(f"\nTop 10 correlated variable pairs for TARGET = {target_val}:")
        for var1, var2, corr in top_corr_pairs:
            print(f"{var1} vs {var2}: {corr:.4f}")

    # Create scatter plots for key relationships
    scatter_pairs = [
        ('AMT_INCOME_TOTAL', 'AMT_CREDIT'),
        ('AMT_INCOME_TOTAL', 'AMT_ANNUITY'),
        ('AMT_CREDIT', 'AMT_ANNUITY'),
        ('DAYS_BIRTH', 'AMT_INCOME_TOTAL')
    ]

    for var1, var2 in scatter_pairs:
        fig = px.scatter(
            df,
            x=var1,
            y=var2,
            color='TARGET',
            opacity=0.6,
            marginal_x='histogram',
            marginal_y='histogram',
            title=f'{var1} vs {var2} by Target',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )

        scatter_figures.append(fig)

    return correlation_figures, scatter_figures

def identify_key_drivers(df):
    """
    Identify key drivers of default

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data

    Returns:
    --------
    list
        List of key drivers and their impact
    """
    key_drivers = []

    # 1. External source scores
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            # Calculate mean scores by target
            mean_scores = df.groupby('TARGET')[col].mean()
            if not mean_scores.isna().any():
                diff_pct = round(((mean_scores[0] - mean_scores[1]) / mean_scores[0] * 100), 2)
                key_drivers.append(f"Low {col} scores increase default risk (Non-defaulters have {diff_pct}% higher scores)")

    # 2. Income to credit ratio
    if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
        df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        mean_ratio = df.groupby('TARGET')['INCOME_TO_CREDIT_RATIO'].mean()
        diff_pct = round(((mean_ratio[0] - mean_ratio[1]) / mean_ratio[0] * 100), 2)
        key_drivers.append(f"Lower income-to-credit ratio increases default risk (Non-defaulters have {diff_pct}% higher ratio)")

    # 3. Age
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
        mean_age = df.groupby('TARGET')['AGE_YEARS'].mean()
        diff_years = round((mean_age[0] - mean_age[1]), 2)
        key_drivers.append(f"Younger clients have higher default risk (Non-defaulters are {diff_years} years older on average)")

    # 4. Employment duration
    if 'DAYS_EMPLOYED' in df.columns:
        # Filter out anomalous values
        df_filtered = df[df['DAYS_EMPLOYED'] < 365*100]
        df_filtered['EMPLOYMENT_YEARS'] = abs(df_filtered['DAYS_EMPLOYED']) / 365.25
        mean_emp = df_filtered.groupby('TARGET')['EMPLOYMENT_YEARS'].mean()
        diff_years = round((mean_emp[0] - mean_emp[1]), 2)
        key_drivers.append(f"Shorter employment duration increases default risk (Non-defaulters have {diff_years} years longer employment)")

    # 5. Gender
    if 'CODE_GENDER' in df.columns:
        gender_default_rate = df.groupby('CODE_GENDER')['TARGET'].mean() * 100
        for gender, rate in gender_default_rate.items():
            if gender != 'Unknown':
                key_drivers.append(f"{gender} gender has {rate:.2f}% default rate")

    return key_drivers

def suggest_strategies(df, key_drivers):
    """
    Suggest actionable strategies based on key drivers

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    key_drivers : list
        List of key drivers

    Returns:
    --------
    list
        List of actionable strategies
    """
    strategies = []

    # 1. External source scores
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            # Find threshold with good separation
            threshold = df[df['TARGET'] == 0][col].quantile(0.25)
            strategies.append(f"Reject applicants with {col} < {threshold:.3f}")

    # 2. Income to credit ratio
    if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
        df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        threshold = df[df['TARGET'] == 0]['INCOME_TO_CREDIT_RATIO'].quantile(0.25)
        strategies.append(f"Implement stricter requirements for applicants with income-to-credit ratio < {threshold:.3f}")

    # 3. Age and employment
    if 'DAYS_BIRTH' in df.columns and 'DAYS_EMPLOYED' in df.columns:
        df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
        df_filtered = df[df['DAYS_EMPLOYED'] < 365*100]
        df_filtered['EMPLOYMENT_YEARS'] = abs(df_filtered['DAYS_EMPLOYED']) / 365.25

        young_threshold = df[df['TARGET'] == 0]['AGE_YEARS'].quantile(0.25)
        emp_threshold = df_filtered[df_filtered['TARGET'] == 0]['EMPLOYMENT_YEARS'].quantile(0.25)

        strategies.append(f"Apply additional verification for applicants younger than {young_threshold:.1f} years with less than {emp_threshold:.1f} years of employment")

    # 4. Gender and car ownership
    if 'CODE_GENDER' in df.columns and 'FLAG_OWN_CAR' in df.columns:
        # Calculate default rates by gender and car ownership
        default_rates = df.groupby(['CODE_GENDER', 'FLAG_OWN_CAR'])['TARGET'].mean() * 100

        for (gender, car), rate in default_rates.items():
            if rate > df['TARGET'].mean() * 100 * 1.2:  # 20% higher than average
                strategies.append(f"Increase interest rates for {gender} applicants with car ownership = '{car}'")

    # 5. Previous application history
    strategies.append("Implement a scoring system that weighs previous application history more heavily")

    return strategies

def run_advanced_eda():
    """
    Run advanced exploratory data analysis

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # 1. Data Loading & Preprocessing
    app_df, prev_df, col_desc, app_df_cleaned, missing_values_summary = load_and_preprocess_data()

    # 2. Outlier Detection
    outlier_columns = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
    outlier_summary, outlier_figures = detect_outliers(app_df_cleaned, outlier_columns)

    # 3. Data Imbalance Check
    imbalance_summary, imbalance_figure = check_data_imbalance(app_df_cleaned)

    # 4. Univariate & Segmented Analysis
    numerical_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH']
    categorical_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']

    # Filter out columns that might not exist
    numerical_cols = [col for col in numerical_cols if col in app_df_cleaned.columns]
    categorical_cols = [col for col in categorical_cols if col in app_df_cleaned.columns]

    numerical_figures, categorical_figures = univariate_analysis(app_df_cleaned, numerical_cols, categorical_cols)

    # 5. Bivariate & Correlation Analysis
    correlation_figures, scatter_figures = bivariate_analysis(app_df_cleaned)

    # 6. Business Insights & Recommendations
    key_drivers = identify_key_drivers(app_df_cleaned)
    strategies = suggest_strategies(app_df_cleaned, key_drivers)

    # Combine results
    results = {
        'app_df': app_df,
        'prev_df': prev_df,
        'col_desc': col_desc,
        'app_df_cleaned': app_df_cleaned,
        'missing_values_summary': missing_values_summary,
        'outlier_summary': outlier_summary,
        'outlier_figures': outlier_figures,
        'imbalance_summary': imbalance_summary,
        'imbalance_figure': imbalance_figure,
        'numerical_figures': numerical_figures,
        'categorical_figures': categorical_figures,
        'correlation_figures': correlation_figures,
        'scatter_figures': scatter_figures,
        'key_drivers': key_drivers,
        'strategies': strategies
    }

    return results

if __name__ == "__main__":
    run_advanced_eda()
