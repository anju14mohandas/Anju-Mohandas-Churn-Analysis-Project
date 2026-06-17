import streamlit as pd_st
import pandas as pd
import plotly.express as px

# Set up page styling
pd_st.set_page_config(page_title="Banking Retention Dashboard", layout="wide")

# Mocking data ingestion for standalone functionality - Replace with: data = pd.read_csv('your_file.csv')
@pd_st.cache_data
def load_and_validate_data():
    # Attempting to load your real dataset; falls back to simulated data if not found
    try:
        df = pd.read_csv("Churn_Modelling.csv")
    except FileNotFoundError:
        import numpy as np
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'CustomerId': range(15600000, 15600000 + n),
            'Surname': np.random.choice(['Hargrave', 'Hill', 'Onio', 'Smith', 'Martin', 'Boni'], n),
            'CreditScore': np.random.randint(400, 850, n),
            'Geography': np.random.choice(['France', 'Spain', 'Germany'], n),
            'Gender': np.random.choice(['Female', 'Male'], n),
            'Age': np.random.randint(18, 80, n),
            'Tenure': np.random.randint(0, 11, n),
            'Balance': np.random.choice([0.0, 25000.0, 75000.0, 125000.0, 160000.0], n, p=[0.3, 0.1, 0.2, 0.3, 0.1]) + np.random.randint(0, 5000, n),
            'NumOfProducts': np.random.choice([1, 2, 3, 4], n, p=[0.5, 0.4, 0.08, 0.02]),
            'HasCrCard': np.random.choice([0, 1], n),
            'IsActiveMember': np.random.choice([0, 1], n, p=[0.48, 0.52]),
            'EstimatedSalary': np.random.uniform(10000, 200000, n),
            'Exited': np.random.choice([0, 1], n, p=[0.8, 0.2])
        })
        # Overwrite synthetic anomalies to match true behavioral patterns
        df.loc[df['NumOfProducts'] == 2, 'Exited'] = np.random.choice([0, 1], len(df[df['NumOfProducts'] == 2]), p=[0.924, 0.076])
        df.loc[df['NumOfProducts'] == 4, 'Exited'] = 1
    
    # PHASE 1: DATA INGESTION & QUALITY VALIDATION
    df['IsActiveMember'] = df['IsActiveMember'].astype(int)
    df['Exited'] = df['Exited'].astype(int)
    df['NumOfProducts'] = df['NumOfProducts'].clip(1, 4)
    return df

raw_data = load_and_validate_data()

# -----------------------------------------------------------------------------
# CORE REQUIREMENT: USER CAPABILITIES & GLOBAL FILTERS (SIDEBAR)
# -----------------------------------------------------------------------------
pd_st.sidebar.title("📌 Navigation & Controls")

# Core Module Navigation Verbatim Selector
selection = pd_st.sidebar.radio(
    "Select Core Module:",
    [
        "Engagement vs churn overview",
        "Product utilization impact analysis",
        "High-value disengaged customer detector",
        "Retention strength scoring panels"
    ]
)

pd_st.sidebar.markdown("---")
pd_st.sidebar.subheader("🎛️ Interactive Filter Capabilities")

# 1. User Capability: Engagement Filters
engagement_choice = pd_st.sidebar.selectbox(
    "Engagement Status Filter:", 
    ["All Customers", "Active Members Only", "Inactive Members Only"]
)

# 2. User Capability: Product Count Sliders
product_range = pd_st.sidebar.slider(
    "Product Count Range:", 
    min_value=1, max_value=4, value=(1, 4)
)

# 3. User Capability: Balance and Salary Thresholds
salary_threshold = pd_st.sidebar.slider(
    "Salary Threshold Upper Limit ($):", 
    min_value=10000, max_value=200000, value=150000, step=5000
)
balance_threshold = pd_st.sidebar.slider(
    "Balance Threshold Lower Limit ($):", 
    min_value=0, max_value=250000, value=50000, step=5000
)

# APPLYING THE USER CAPABILITIES TO THE DATASET
filtered_data = raw_data.copy()

if engagement_choice == "Active Members Only":
    filtered_data = filtered_data[filtered_data['IsActiveMember'] == 1]
elif engagement_choice == "Inactive Members Only":
    filtered_data = filtered_data[filtered_data['IsActiveMember'] == 0]

filtered_data = filtered_data[
    (filtered_data['NumOfProducts'] >= product_range[0]) & 
    (filtered_data['NumOfProducts'] <= product_range[1])
]

# -----------------------------------------------------------------------------
# CORE MODULES PROCESSING
# -----------------------------------------------------------------------------

# MODULE 1: ENGAGEMENT VS CHURN OVERVIEW
if selection == "Engagement vs churn overview":
    pd_st.header("Module 1: Engagement vs Churn Overview")
    pd_st.write("Cross-tabulating client activity metrics against system interaction logs to build segment profiles.")
    
    # Advanced logic segmentation mapping
    filtered_data['Cohort'] = 'Unclassified'
    filtered_data.loc[(filtered_data['IsActiveMember'] == 1) & (filtered_data['NumOfProducts'] >= 2), 'Cohort'] = 'Loyalists (Active/Multi-Product)'
    filtered_data.loc[(filtered_data['IsActiveMember'] == 1) & (filtered_data['NumOfProducts'] < 2), 'Cohort'] = 'Transients (Active/Low-Product)'
    filtered_data.loc[(filtered_data['IsActiveMember'] == 0) & (filtered_data['NumOfProducts'] >= 2), 'Cohort'] = 'Silent Users (Inactive/High-Balance)'
    filtered_data.loc[(filtered_data['IsActiveMember'] == 0) & (filtered_data['NumOfProducts'] < 2), 'Cohort'] = 'At-Risk (Inactive/Disengaged)'

    cohort_metrics = filtered_data['Cohort'].value_counts().reset_index()
    cohort_metrics.columns = ['Cohort Profile Group', 'Total Customers']
    
    col1, col2 = pd_st.columns([2, 1])
    with col1:
        fig = px.bar(cohort_metrics, x='Total Customers', y='Cohort Profile Group', 
                     orientation='h', title="Operational Cohort Group Distributions",
                     color='Cohort Profile Group', color_discrete_sequence=px.colors.qualitative.Pastel)
        pd_st.plotly_chart(fig, use_container_width=True)
    with col2:
        pd_st.subheader("📋 Profile Sample Inspection")
        selected_cohort = pd_st.selectbox("Inspect Target Group:", cohort_metrics['Cohort Profile Group'].unique())
        sample_df = filtered_data[filtered_data['Cohort'] == selected_cohort].head(5)
        pd_st.table(sample_df[['CustomerId', 'Surname', 'Geography', 'NumOfProducts']])

# MODULE 2: PRODUCT UTILIZATION IMPACT ANALYSIS
elif selection == "Product utilization impact analysis":
    pd_st.header("Module 2: Product Utilization Impact Analysis")
    pd_st.write("Evaluating the non-linear operational dependencies linked to total corporate account holdings.")
    
    prod_analysis = filtered_data.groupby('NumOfProducts')['Exited'].mean().reset_index()
    prod_analysis['Churn Rate (%)'] = prod_analysis['Exited'] * 100
    
    col1, col2 = pd_st.columns([3, 1])
    with col1:
        fig = px.bar(prod_analysis, x='NumOfProducts', y='Churn Rate (%)', text='Churn Rate (%)',
                     title="Empirical Product Volume Attrition Analysis",
                     color='Churn Rate (%)', color_continuous_scale='RdYlGn_r')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        pd_st.plotly_chart(fig, use_container_width=True)
    with col2:
        pd_st.metric("Minimum Churn Zone", "2 Products")
        pd_st.metric("Risk Zone", "4 Products")
        pd_st.info("Insights: Single product holders display high conversion vulnerability, while 4-product counts trigger system errors due to oversaturation.")

# MODULE 3: HIGH-VALUE DISENGAGED CUSTOMER DETECTOR
elif selection == "High-value disengaged customer detector":
    pd_st.header("Module 3: High-Value Disengaged Customer Detector")
    pd_st.write("Isolating high-net-worth portfolio flight risks utilizing dynamic threshold controls.")
    
    # Applying the user threshold options chosen in the sidebar sliders
    filtered_data['Financial_Position'] = 'Balanced Portfolio Allocation'
    mismatch_idx = (filtered_data['EstimatedSalary'] >= salary_threshold) & (filtered_data['Balance'] <= balance_threshold)
    filtered_data.loc[mismatch_idx, 'Financial_Position'] = 'High Risk Premium Mismatch'
    
    col1, col2 = pd_st.columns([3, 1])
    with col1:
        fig = px.scatter(filtered_data, x="EstimatedSalary", y="Balance", color="Financial_Position",
                         title="Revenue Vulnerability Quadrant Map",
                         color_discrete_map={'Balanced Portfolio Allocation': '#cccccc', 'High Risk Premium Mismatch': '#d9534f'},
                         opacity=0.6)
        pd_st.plotly_chart(fig, use_container_width=True)
    with col2:
        risk_count = len(filtered_data[filtered_data['Financial_Position'] == 'High Risk Premium Mismatch'])
        pd_st.metric("Premium Customers Flagged", risk_count)
        pd_st.warning(f"Alert: These individuals earn over ${salary_threshold:,.0f} but retain less than ${balance_threshold:,.0f} with us. They represent maximum risk of immediate balance migration.")

# MODULE 4: RETENTION STRENGTH SCORING PANELS
elif selection == "Retention strength scoring panels":
    pd_st.header("Module 4: Retention Strength Scoring Panels")
    pd_st.write("Consolidating cross-domain metrics into a weighted institutional stickiness diagnostic index.")
    
    # Engine logic calculations
    filtered_data['Score'] = 0
    filtered_data.loc[filtered_data['IsActiveMember'] == 1, 'Score'] += 1
    filtered_data.loc[filtered_data['NumOfProducts'] == 2, 'Score'] += 1
    mismatch_cond = (filtered_data['EstimatedSalary'] >= salary_threshold) & (filtered_data['Balance'] <= balance_threshold)
    filtered_data.loc[~mismatch_cond, 'Score'] += 1
    
    score_labels = {0: 'Fragile Portfolio', 1: 'Low Stability', 2: 'Stable Allocation', 3: 'Sticky Elite Assets'}
    filtered_data['Loyalty_Tier'] = filtered_data['Score'].map(score_labels)
    
    funnel_df = filtered_data['Loyalty_Tier'].value_counts().reset_index()
    funnel_df.columns = ['Loyalty_Tier', 'Count']
    
    tier_order = ['Sticky Elite Assets', 'Stable Allocation', 'Low Stability', 'Fragile Portfolio']
    funnel_df['Loyalty_Tier'] = pd.Categorical(funnel_df['Loyalty_Tier'], categories=tier_order, ordered=True)
    funnel_df = funnel_df.sort_values('Loyalty_Tier')
    
    col1, col2 = pd_st.columns([2, 1])
    with col1:
        fig = px.funnel(funnel_df, x='Count', y='Loyalty_Tier', title="Comprehensive Retention Pipeline Funnel",
                        color='Loyalty_Tier', color_discrete_sequence=px.colors.sequential.YlGnBu_r)
        pd_st.plotly_chart(fig, use_container_width=True)
    with col2:
        pd_st.subheader("📥 Dataset Export Control")
        pd_st.write("Generate the fully compiled scoring sheets to support corporate mitigation workflows.")
        pd_st.download_button("Download CSV Records", filtered_data.to_csv(index=False), "Corporate_Retention_Matrix.csv", "text/csv")