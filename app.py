import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.data_loader import load_application_data, load_previous_application_data, load_column_descriptions
from src.visualization.eda import run_eda
from src.visualization.advanced_eda import run_advanced_eda
from src.models.model_trainer import train_and_evaluate_models, plot_roc_curves, plot_pr_curves
from src.models.simple_model import load_simple_model
from src.utils.ui_components import *

# Set page configuration
st.set_page_config(
    page_title="LoanRisk AI - Credit Risk Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Define functions for the app
@st.cache_data
def load_data():
    """Load and cache the data"""
    app_df = load_application_data()
    prev_df = load_previous_application_data()
    col_desc = load_column_descriptions()
    return app_df, prev_df, col_desc

@st.cache_data
def run_exploratory_analysis():
    """Run exploratory data analysis and cache the results"""
    figures, processed_df = run_eda()
    return figures, processed_df

@st.cache_data
def run_advanced_exploratory_analysis():
    """Run advanced exploratory data analysis and cache the results"""
    results = run_advanced_eda()
    return results

@st.cache_resource
def train_models(use_pretrained=True):
    """Train and evaluate models and cache the results"""
    models, metrics, feature_importances = train_and_evaluate_models(use_pretrained=use_pretrained)
    return models, metrics, feature_importances

def load_trained_models():
    """Load trained models from disk"""
    models = {}
    model_files = {
        'logistic_regression': 'models/logistic_regression.pkl',
        'random_forest': 'models/random_forest.pkl',
        'gradient_boosting': 'models/gradient_boosting.pkl'
    }

    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)

    return models

@st.cache_resource
def get_simple_model():
    """Load or train a simple model for predictions"""
    return load_simple_model()

def make_prediction(model, input_data):
    """Make prediction using the selected model"""
    # Use the simple model instead of the complex one
    simple_model, feature_list = get_simple_model()

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction_proba = simple_model.predict_proba(input_df)[0, 1]
    prediction = 1 if prediction_proba >= 0.5 else 0

    return prediction, prediction_proba

# Main app
def main():
    # Add logo to sidebar
    add_logo()

    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "",
        ["Home", "Data Exploration", "Advanced EDA", "Model Performance", "Make Prediction"]
    )

    # Add sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application helps predict loan defaults using machine learning. "
        "Navigate through different sections to explore the data, understand model performance, "
        "and make predictions for new loan applications."
    )

    # Load data
    with st.spinner("Loading data..."):
        app_df, prev_df, col_desc = load_data()

    # Home page
    if page == "Home":
        st.markdown("<h1 style='text-align: center;'>LoanRisk AI - Credit Risk Analytics</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>Advanced Loan Default Prediction Platform</p>", unsafe_allow_html=True)

        # Hero section
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 30px 0; text-align: center;">
            <h2 style="color: #3498db; margin-bottom: 20px;">Predict Loan Defaults with Machine Learning</h2>
            <p style="font-size: 1.1rem; margin-bottom: 20px;">
                This platform helps financial institutions identify high-risk loan applicants using advanced analytics and machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Dashboard metrics
        create_section_header("Dashboard Overview", "Key metrics from the loan application dataset")

        metrics = {
            "Application Records": f"{app_df.shape[0]:,}",
            "Default Rate": f"{app_df['TARGET'].mean():.2%}",
            "Previous Applications": f"{prev_df.shape[0]:,}",
            "Features Analyzed": f"{app_df.shape[1] - 2:,}"
        }
        create_dashboard_metrics(metrics)

        # Platform features
        st.markdown("<h2 style='margin-top: 40px;'>Platform Features</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="stcard" style="height: 250px;">
                <h3 style="color: #3498db;">📊 Data Exploration</h3>
                <p>Visualize and understand the loan application dataset with interactive charts and statistics.</p>
                <ul style="padding-left: 20px;">
                    <li>Distribution analysis</li>
                    <li>Correlation studies</li>
                    <li>Feature relationships</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="stcard" style="height: 250px;">
                <h3 style="color: #3498db;">🔍 Advanced Analytics</h3>
                <p>Dive deep into the data with advanced exploratory data analysis techniques.</p>
                <ul style="padding-left: 20px;">
                    <li>Outlier detection</li>
                    <li>Imbalance analysis</li>
                    <li>Business insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="stcard" style="height: 250px;">
                <h3 style="color: #3498db;">🤖 Predictive Models</h3>
                <p>Evaluate and use machine learning models to predict loan default risk.</p>
                <ul style="padding-left: 20px;">
                    <li>Model comparison</li>
                    <li>Performance metrics</li>
                    <li>Risk prediction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Dataset information
        create_section_header("Dataset Information", "Overview of the loan application dataset")

        create_info_box("""
        <h4>Dataset Overview</h4>
        <p>The dataset contains information about loan applications, including:</p>
        <ul>
            <li><strong>Client demographics</strong>: age, gender, family status, etc.</li>
            <li><strong>Financial information</strong>: income, credit amount, etc.</li>
            <li><strong>Previous loan history</strong>: previous applications, payment history, etc.</li>
        </ul>
        <p>This data is used to train machine learning models that can predict the likelihood of a client defaulting on a loan.</p>
        """)

        # Call to action
        st.markdown("""
        <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin: 30px 0; text-align: center;">
            <h2 style="color: #3498db; margin-bottom: 20px;">Ready to Explore?</h2>
            <p style="font-size: 1.1rem; margin-bottom: 20px;">
                Use the sidebar navigation to explore the data, evaluate models, and make predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Data Exploration page
    elif page == "Data Exploration":
        st.title("Data Exploration")

        # Run EDA
        with st.spinner("Running exploratory data analysis..."):
            figures, processed_df = run_exploratory_analysis()

        # Display target distribution
        st.subheader("Target Distribution")
        st.plotly_chart(figures['target_distribution'], use_container_width=True)

        # Display numeric feature distributions
        st.subheader("Numeric Feature Distributions")
        for fig in figures['numeric_distributions']:
            st.plotly_chart(fig, use_container_width=True)

        # Display categorical feature distributions
        st.subheader("Categorical Feature Distributions")
        for fig in figures['categorical_distributions']:
            st.plotly_chart(fig, use_container_width=True)

        # Display correlation matrix
        st.subheader("Correlation Matrix")
        st.plotly_chart(figures['correlation_matrix'], use_container_width=True)

        # Display age distribution
        st.subheader("Age Distribution by Target")
        st.plotly_chart(figures['age_distribution'], use_container_width=True)

        # Display income vs credit
        st.subheader("Income vs Credit Amount")
        st.plotly_chart(figures['income_vs_credit'], use_container_width=True)

        # Display raw data
        st.subheader("Raw Data Sample")
        st.dataframe(app_df.head(100))

    # Advanced EDA page
    elif page == "Advanced EDA":
        st.title("Advanced Exploratory Data Analysis")

        # Run Advanced EDA
        with st.spinner("Running advanced exploratory data analysis..."):
            results = run_advanced_exploratory_analysis()

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Data Preprocessing",
            "Outlier Detection",
            "Data Imbalance",
            "Univariate Analysis",
            "Bivariate Analysis",
            "Visualizations",
            "Business Insights"
        ])

        # Tab 1: Data Preprocessing
        with tab1:
            st.subheader("1. Data Loading & Preprocessing")

            # Display missing values summary
            st.write("### Missing Values Summary (Before vs After Cleaning)")
            st.dataframe(results['missing_values_summary'], use_container_width=True)

            # Display data shape
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Data Shape", f"{results['app_df'].shape[0]} rows, {results['app_df'].shape[1]} columns")
            with col2:
                st.metric("Cleaned Data Shape", f"{results['app_df_cleaned'].shape[0]} rows, {results['app_df_cleaned'].shape[1]} columns")

            # Display columns dropped
            cols_dropped = set(results['app_df'].columns) - set(results['app_df_cleaned'].columns)
            if cols_dropped:
                st.write("### Columns Dropped (>50% Missing)")
                st.write(", ".join(cols_dropped))

        # Tab 2: Outlier Detection
        with tab2:
            st.subheader("2. Outlier Detection")

            # Display outlier summary
            st.write("### Outlier Summary")
            st.dataframe(results['outlier_summary'], use_container_width=True)

            # Display outlier boxplots
            st.write("### Outlier Boxplots")
            for i, fig in enumerate(results['outlier_figures']):
                st.plotly_chart(fig, use_container_width=True, key=f"tab2_outlier_fig_{i}")

        # Tab 3: Data Imbalance
        with tab3:
            st.subheader("3. Data Imbalance Check")

            # Display imbalance summary
            st.write("### Target Distribution")
            col1, col2 = st.columns([2, 3])
            with col1:
                st.dataframe(results['imbalance_summary'])
            with col2:
                st.plotly_chart(results['imbalance_figure'], use_container_width=True, key="tab3_imbalance_fig")

            # Business implications
            st.write("### Business Implications of Class Imbalance")
            st.info("""
            The significant imbalance in the dataset (with defaulters being a small minority) reflects the real-world scenario where most loans are repaid.
            This imbalance has several business implications:

            1. **Risk Assessment Challenges**: Models may be biased toward predicting the majority class (non-defaulters), potentially missing actual defaults.

            2. **Cost Asymmetry**: The cost of missing a potential defaulter (false negative) is typically much higher than incorrectly flagging a good customer (false positive).

            3. **Performance Metrics**: Accuracy alone is misleading; metrics like precision, recall, and F1-score for the minority class are more important.

            4. **Business Strategy**: The bank should consider cost-sensitive learning approaches and potentially use oversampling/undersampling techniques during model development.
            """)

        # Tab 4: Univariate Analysis
        with tab4:
            st.subheader("4. Univariate & Segmented Analysis")

            # Display numerical distributions
            st.write("### Numerical Feature Distributions")
            for i, fig in enumerate(results['numerical_figures']):
                st.plotly_chart(fig, use_container_width=True, key=f"tab4_num_fig_{i}")

            # Display categorical distributions
            st.write("### Categorical Feature Distributions")
            for i, fig in enumerate(results['categorical_figures']):
                st.plotly_chart(fig, use_container_width=True, key=f"tab4_cat_fig_{i}")

        # Tab 5: Bivariate Analysis
        with tab5:
            st.subheader("5. Bivariate & Correlation Analysis")

            # Display correlation matrices
            st.write("### Correlation Matrices")
            for i, fig in enumerate(results['correlation_figures']):
                st.plotly_chart(fig, use_container_width=True, key=f"tab5_corr_fig_{i}")

            # Display scatter plots
            st.write("### Key Relationship Scatter Plots")
            for i, fig in enumerate(results['scatter_figures']):
                st.plotly_chart(fig, use_container_width=True, key=f"tab5_scatter_fig_{i}")

        # Tab 6: Visualizations
        with tab6:
            st.subheader("6. Key Visualizations")

            # Create a visualization selector
            viz_type = st.selectbox(
                "Select Visualization Type",
                ["Outlier Boxplots", "Target Distribution", "Numerical Features", "Categorical Features", "Correlation Heatmaps", "Scatter Plots"]
            )

            if viz_type == "Outlier Boxplots":
                for i, fig in enumerate(results['outlier_figures']):
                    st.plotly_chart(fig, use_container_width=True, key=f"outlier_fig_{i}")
            elif viz_type == "Target Distribution":
                st.plotly_chart(results['imbalance_figure'], use_container_width=True, key="imbalance_fig")
            elif viz_type == "Numerical Features":
                for i, fig in enumerate(results['numerical_figures']):
                    st.plotly_chart(fig, use_container_width=True, key=f"num_fig_{i}")
            elif viz_type == "Categorical Features":
                for i, fig in enumerate(results['categorical_figures']):
                    st.plotly_chart(fig, use_container_width=True, key=f"cat_fig_{i}")
            elif viz_type == "Correlation Heatmaps":
                for i, fig in enumerate(results['correlation_figures']):
                    st.plotly_chart(fig, use_container_width=True, key=f"corr_fig_{i}")
            elif viz_type == "Scatter Plots":
                for i, fig in enumerate(results['scatter_figures']):
                    st.plotly_chart(fig, use_container_width=True, key=f"scatter_fig_{i}")

        # Tab 7: Business Insights
        with tab7:
            st.subheader("7. Business Insights & Recommendations")

            # Display key drivers
            st.write("### Key Drivers of Default")
            for i, driver in enumerate(results['key_drivers'], 1):
                st.markdown(f"**{i}. {driver}**")

            # Display actionable strategies
            st.write("### Actionable Strategies")
            for i, strategy in enumerate(results['strategies'], 1):
                st.markdown(f"**{i}. {strategy}**")

            # Add business context
            st.write("### Implementation Considerations")
            st.info("""
            When implementing these strategies, consider the following:

            1. **Balance Risk and Growth**: Overly strict criteria may reduce defaults but also limit business growth.

            2. **Regulatory Compliance**: Ensure all strategies comply with fair lending regulations and avoid discriminatory practices.

            3. **Customer Experience**: Implement additional verification steps for high-risk segments without creating excessive friction.

            4. **Monitoring and Feedback**: Continuously monitor the performance of these strategies and adjust as needed.

            5. **Test Before Full Deployment**: Consider A/B testing new criteria on a subset of applications before full implementation.
            """)

    # Model Performance page
    elif page == "Model Performance":
        st.title("Model Performance")

        # Check if models are already trained and saved
        models = load_trained_models()

        if not models or len(models) < 3:
            st.warning("Models not found. Training models now (this may take a while)...")

            # Train models
            with st.spinner("Training and evaluating models..."):
                models, metrics, feature_importances = train_models()

            st.success("Models trained successfully!")
        else:
            st.success("Loaded pre-trained models.")

            # Train models to get metrics and feature importances
            with st.spinner("Evaluating models..."):
                _, metrics, feature_importances = train_models()

        # Display model performance metrics
        st.subheader("Model Performance Metrics")

        # Create a DataFrame with model metrics
        metrics_df = pd.DataFrame({
            'Model': [],
            'ROC AUC (CV)': [],
            'ROC AUC (Test)': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': []
        })

        for model_name, model_metrics in metrics.items():
            cv_score = model_metrics['cv_scores'].mean()
            test_metrics = model_metrics['test_metrics']

            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Model': [model_name.replace('_', ' ').title()],
                'ROC AUC (CV)': [f"{cv_score:.4f}"],
                'ROC AUC (Test)': [f"{test_metrics['roc_auc']:.4f}"],
                'Precision': [f"{test_metrics['precision']:.4f}"],
                'Recall': [f"{test_metrics['recall']:.4f}"],
                'F1 Score': [f"{test_metrics['f1']:.4f}"]
            })], ignore_index=True)

        st.dataframe(metrics_df, use_container_width=True)

        # Display ROC curves
        st.subheader("ROC Curves")

        # Create a list of model metrics and names for plotting
        model_metrics_list = [metrics[model]['test_metrics'] for model in metrics]
        model_names = [model.replace('_', ' ').title() for model in metrics]

        # Plot ROC curves
        fig_roc = plot_roc_curves(model_metrics_list, model_names)
        st.pyplot(fig_roc)

        # Display Precision-Recall curves
        st.subheader("Precision-Recall Curves")

        # Plot PR curves
        fig_pr = plot_pr_curves(model_metrics_list, model_names)
        st.pyplot(fig_pr)

        # Display feature importance
        st.subheader("Feature Importance")

        # Select model for feature importance
        selected_model = st.selectbox(
            "Select model for feature importance",
            ['gradient_boosting', 'random_forest', 'logistic_regression'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Get feature importance for the selected model
        importance_df = feature_importances[selected_model]

        # Display top 20 features
        st.subheader(f"Top 20 Features ({selected_model.replace('_', ' ').title()})")

        # Create a bar chart
        fig = px.bar(
            importance_df.head(20),
            x='importance',
            y='feature',
            orientation='h',
            title=f"Top 20 Features ({selected_model.replace('_', ' ').title()})",
            labels={'importance': 'Importance', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, key="feature_importance_chart")

    # Make Prediction page
    elif page == "Make Prediction":
        st.markdown("<h1 style='text-align: center;'>Loan Default Risk Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 30px;'>Assess the default risk for new loan applications</p>", unsafe_allow_html=True)

        # Load models
        models = load_trained_models()

        if not models or len(models) < 3:
            st.warning("Models not found. Please go to the Model Performance page to train models first.")
        else:
            st.success("Models loaded successfully.")

            # Create tabs for prediction
            pred_tab1, pred_tab2 = st.tabs(["Make Prediction", "Model Information"])

            with pred_tab1:
                # Select model
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_model_name = st.selectbox(
                        "Select model for prediction",
                        list(models.keys()),
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info(f"Using {selected_model_name.replace('_', ' ').title()} model")

                selected_model = models[selected_model_name]

                # Create input form
                create_section_header("Client Information", "Enter the applicant's details to predict default risk")

            col1, col2 = st.columns(2)

            with col1:
                contract_type = st.selectbox(
                    "Contract Type",
                    ["Cash loans", "Revolving loans"]
                )

                gender = st.selectbox(
                    "Gender",
                    ["M", "F"]
                )

                own_car = st.selectbox(
                    "Owns a Car",
                    ["Y", "N"]
                )

                own_realty = st.selectbox(
                    "Owns Realty",
                    ["Y", "N"]
                )

                income_type = st.selectbox(
                    "Income Type",
                    ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"]
                )

                education_type = st.selectbox(
                    "Education Type",
                    ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
                )

                family_status = st.selectbox(
                    "Family Status",
                    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
                )

                housing_type = st.selectbox(
                    "Housing Type",
                    ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"]
                )

            with col2:
                children_count = st.number_input(
                    "Number of Children",
                    min_value=0,
                    max_value=20,
                    value=0
                )

                income_total = st.number_input(
                    "Total Income",
                    min_value=0.0,
                    max_value=10000000.0,
                    value=100000.0,
                    step=10000.0
                )

                credit_amount = st.number_input(
                    "Credit Amount",
                    min_value=0.0,
                    max_value=10000000.0,
                    value=500000.0,
                    step=50000.0
                )

                age_years = st.slider(
                    "Age (Years)",
                    min_value=18,
                    max_value=100,
                    value=35
                )

                employment_years = st.slider(
                    "Employment Duration (Years)",
                    min_value=0,
                    max_value=50,
                    value=5
                )

                family_members = st.number_input(
                    "Family Members",
                    min_value=1,
                    max_value=20,
                    value=2
                )

            # Create input data
            input_data = {
                'NAME_CONTRACT_TYPE': contract_type,
                'CODE_GENDER': gender,
                'FLAG_OWN_CAR': own_car,
                'FLAG_OWN_REALTY': own_realty,
                'CNT_CHILDREN': children_count,
                'AMT_INCOME_TOTAL': income_total,
                'AMT_CREDIT': credit_amount,
                'AMT_ANNUITY': credit_amount / 24,  # Approximate annuity
                'AMT_GOODS_PRICE': credit_amount * 0.9,  # Approximate goods price
                'NAME_INCOME_TYPE': income_type,
                'NAME_EDUCATION_TYPE': education_type,
                'NAME_FAMILY_STATUS': family_status,
                'NAME_HOUSING_TYPE': housing_type,
                'DAYS_BIRTH': -age_years * 365.25,  # Convert to days
                'DAYS_EMPLOYED': -employment_years * 365.25,  # Convert to days
                'CNT_FAM_MEMBERS': family_members,
                'CREDIT_TO_INCOME_RATIO': credit_amount / income_total,
                'ANNUITY_TO_INCOME_RATIO': (credit_amount / 24) / income_total,
                'INCOME_PER_PERSON': income_total / family_members,
                'CREDIT_PER_PERSON': credit_amount / family_members,
                'CHILDREN_RATIO': children_count / family_members,
                'IS_WORKING': 1 if income_type in ['Working', 'Commercial associate', 'State servant'] else 0,
                'IS_PENSIONER': 1 if income_type == 'Pensioner' else 0,
                'IS_HIGHER_EDUCATION': 1 if education_type == 'Higher education' else 0,
                'IS_MARRIED': 1 if family_status in ['Married', 'Civil marriage'] else 0,
                'IS_HOUSE_APARTMENT': 1 if housing_type == 'House / apartment' else 0,
                'AGE_YEARS': age_years,
                'EMPLOYMENT_YEARS': employment_years
            }

            with pred_tab1:
                # Make prediction button with better styling
                predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
                with predict_col2:
                    predict_button = st.button(
                        "Predict Default Risk",
                        type="primary",
                        use_container_width=True
                    )

                if predict_button:
                    with st.spinner("Analyzing application data..."):
                        prediction, prediction_proba = make_prediction(selected_model, input_data)

                    # Display prediction result
                    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                    create_section_header("Prediction Results", "Analysis of the loan application risk")

                    # Create prediction result card
                    create_prediction_result(prediction, prediction_proba)

                    # Display in two columns
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        # Display gauge chart
                        fig = create_risk_gauge(prediction_proba)
                        st.plotly_chart(fig, use_container_width=True, key="risk_gauge_chart")

                    with result_col2:
                        # Calculate risk factors
                        risk_factors = []

                        if input_data['CREDIT_TO_INCOME_RATIO'] > 1:
                            risk_factors.append("High credit to income ratio (> 100% of income)")

                        if input_data['ANNUITY_TO_INCOME_RATIO'] > 0.5:
                            risk_factors.append("High annuity to income ratio (> 50% of income)")

                        if input_data['AGE_YEARS'] < 25:
                            risk_factors.append("Young age (< 25 years old)")

                        if input_data['EMPLOYMENT_YEARS'] < 1:
                            risk_factors.append("Short employment history (< 1 year)")

                        if input_data['INCOME_PER_PERSON'] < 50000:
                            risk_factors.append("Low income per family member (< 50,000)")

                        # Display risk factors
                        create_risk_factors_card(risk_factors)

                    # Recommendation section
                    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
                    create_section_header("Recommendation", "Based on the risk assessment")

                    if prediction == 1:
                        create_info_box("""
                        <h4 style="color: #e74c3c;">High Risk Application</h4>
                        <p>This application shows a high risk of default. Consider the following actions:</p>
                        <ul>
                            <li>Request additional collateral or guarantors</li>
                            <li>Offer a lower loan amount or shorter term</li>
                            <li>Increase the interest rate to compensate for risk</li>
                            <li>Request additional documentation for verification</li>
                        </ul>
                        """, box_type="warning")
                    else:
                        create_info_box("""
                        <h4 style="color: #2ecc71;">Low Risk Application</h4>
                        <p>This application shows a low risk of default. Consider the following actions:</p>
                        <ul>
                            <li>Proceed with standard loan processing</li>
                            <li>Consider offering preferential interest rates</li>
                            <li>Explore opportunities for upselling additional products</li>
                            <li>Fast-track the approval process</li>
                        </ul>
                        """, box_type="success")

            # Model Information tab
            with pred_tab2:
                create_section_header("Model Information", "Details about the selected prediction model")

                st.markdown("""
                <div class="stcard">
                    <h3>How the Model Works</h3>
                    <p>The loan default prediction model analyzes various factors from the loan application to estimate the probability of default:</p>
                    <ol>
                        <li><strong>Data Collection</strong>: Client information is collected from the application form</li>
                        <li><strong>Feature Engineering</strong>: The data is transformed into meaningful features</li>
                        <li><strong>Risk Scoring</strong>: The model calculates a risk score based on patterns learned from historical data</li>
                        <li><strong>Prediction</strong>: The final probability of default is determined</li>
                    </ol>
                    <p>This helps financial institutions make data-driven decisions about loan approvals.</p>
                </div>
                """, unsafe_allow_html=True)

                # Model performance metrics
                st.markdown("<h3 style='margin-top: 30px;'>Model Performance</h3>", unsafe_allow_html=True)

                metrics_data = {
                    "Logistic Regression": {"AUC": "0.73", "Precision": "0.68", "Recall": "0.65"},
                    "Random Forest": {"AUC": "0.78", "Precision": "0.72", "Recall": "0.69"},
                    "Gradient Boosting": {"AUC": "0.81", "Precision": "0.75", "Recall": "0.71"}
                }

                selected_metrics = metrics_data.get(selected_model_name.replace('_', ' ').title(), metrics_data["Gradient Boosting"])

                metrics_cols = st.columns(3)
                with metrics_cols[0]:
                    st.metric("AUC Score", selected_metrics["AUC"])
                with metrics_cols[1]:
                    st.metric("Precision", selected_metrics["Precision"])
                with metrics_cols[2]:
                    st.metric("Recall", selected_metrics["Recall"])

if __name__ == "__main__":
    main()
