import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64

def load_css():
    """Load custom CSS"""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def add_logo():
    """Add logo to the sidebar"""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0 30px;">
            <h1 style="color: #3498db; margin-bottom: 0;">LoanRisk</h1>
            <p style="color: #7f8c8d; margin-top: 0;">AI-Powered Credit Risk Analytics</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def create_metric_card(title, value, description=None, delta=None, delta_color="normal"):
    """Create a custom metric card"""
    st.markdown(
        f"""
        <div class="stcard">
            <h3>{title}</h3>
            <p style="font-size: 2rem; font-weight: 700; margin: 10px 0; color: #3498db;">{value}</p>
            {f'<p style="color: #7f8c8d; margin: 0;">{description}</p>' if description else ''}
            {f'<p style="color: {"#2ecc71" if delta > 0 else "#e74c3c"}; font-weight: 500;">{"+" if delta > 0 else ""}{delta}%</p>' if delta is not None else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_dashboard_metrics(metrics_dict):
    """Create a row of dashboard metrics"""
    cols = st.columns(len(metrics_dict))
    for i, (title, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(title, value)

def create_feature_importance_chart(importance_df, top_n=10):
    """Create a feature importance chart"""
    # Take top N features
    top_features = importance_df.head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Important Features',
        labels={'importance': 'Importance', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif", size=12),
        title_font=dict(family="Roboto, sans-serif", size=18),
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=False)
    )
    
    return fig

def create_risk_gauge(risk_score):
    """Create a risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Risk", 'font': {'size': 24, 'family': 'Roboto, sans-serif'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': "#2ecc71"},  # Green
                {'range': [25, 50], 'color': "#f1c40f"},  # Yellow
                {'range': [50, 75], 'color': "#e67e22"},  # Orange
                {'range': [75, 100], 'color': "#e74c3c"}  # Red
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Roboto, sans-serif", size=12)
    )
    
    return fig

def create_risk_factors_card(risk_factors):
    """Create a card with risk factors"""
    if not risk_factors:
        st.info("No significant risk factors identified")
        return
    
    st.markdown(
        """
        <div class="stcard">
            <h3>Risk Factors</h3>
            <ul style="padding-left: 20px;">
        """ + 
        "".join([f'<li style="margin-bottom: 8px;">{factor}</li>' for factor in risk_factors]) +
        """
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_prediction_result(prediction, prediction_proba):
    """Create a prediction result card"""
    risk_level = "High" if prediction == 1 else "Low"
    risk_class = "risk-high" if prediction == 1 else "risk-low"
    
    st.markdown(
        f"""
        <div class="stcard" style="text-align: center;">
            <h2>Prediction Result</h2>
            <p style="font-size: 1.5rem; margin: 20px 0;">
                This client has a <span class="{risk_class}">{risk_level} Risk</span> of default
            </p>
            <p style="font-size: 2.5rem; font-weight: 700; margin: 20px 0; color: {"#e74c3c" if prediction == 1 else "#2ecc71"};">
                {prediction_proba:.1%}
            </p>
            <p style="color: #7f8c8d;">Default Probability</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_section_header(title, description=None):
    """Create a section header with optional description"""
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <h2 style="color: #2c3e50; margin-bottom: 5px;">{title}</h2>
            {f'<p style="color: #7f8c8d;">{description}</p>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_info_box(content, box_type="info"):
    """Create an info box with custom styling"""
    colors = {
        "info": "#3498db",
        "success": "#2ecc71",
        "warning": "#f39c12",
        "error": "#e74c3c"
    }
    
    st.markdown(
        f"""
        <div style="background-color: {colors[box_type]}10; border-left: 5px solid {colors[box_type]}; padding: 15px; border-radius: 4px; margin: 20px 0;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )

def create_comparison_table(df1, df2, title1, title2):
    """Create a comparison table between two dataframes"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='text-align: center;'>{title1}</h3>", unsafe_allow_html=True)
        st.dataframe(df1)
    
    with col2:
        st.markdown(f"<h3 style='text-align: center;'>{title2}</h3>", unsafe_allow_html=True)
        st.dataframe(df2)

def create_animated_counter(value, prefix="", suffix="", duration=1000):
    """Create an animated counter"""
    st.markdown(
        f"""
        <div class="counter" data-target="{value}" data-prefix="{prefix}" data-suffix="{suffix}">
            <span class="counter-value">{prefix}0{suffix}</span>
        </div>
        <script>
            const counters = document.querySelectorAll('.counter');
            counters.forEach(counter => {{
                const target = +counter.getAttribute('data-target');
                const prefix = counter.getAttribute('data-prefix');
                const suffix = counter.getAttribute('data-suffix');
                const duration = {duration};
                const increment = target / (duration / 16);
                let current = 0;
                
                const updateCounter = () => {{
                    current += increment;
                    if (current < target) {{
                        counter.querySelector('.counter-value').textContent = `${{prefix}}${{Math.floor(current)}}${{suffix}}`;
                        setTimeout(updateCounter, 16);
                    }} else {{
                        counter.querySelector('.counter-value').textContent = `${{prefix}}${{target}}${{suffix}}`;
                    }}
                }};
                
                updateCounter();
            }});
        </script>
        """,
        unsafe_allow_html=True
    )
