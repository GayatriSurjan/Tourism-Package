# ======================================================
# HR ATTRITION PREDICTION WEB APPLICATION
# Senior Lead Data Scientist Version
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

# ------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Intelligence System",
    layout="wide"
)

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("hr_dataset.csv")

df = load_data()

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
st.sidebar.title("Navigation")

phase = st.sidebar.radio(
    "Select Phase",
    [
        "Phase 1: Dataset Overview",
        "Phase 2: Univariate Analysis",
        "Phase 3: Bivariate Analysis",
        "Phase 4: Multivariate Analysis",
        "Phase 5: Machine Learning Predictor"
    ]
)

# ======================================================
# PHASE 1 — DATASET OVERVIEW
# ======================================================
if phase == "Phase 1: Dataset Overview":

    st.title("📊 Dataset Overview")

    # ----------------------------
    # KPI SECTION
    # ----------------------------
    col1, col2, col3, col4 = st.columns(4)

    total_rows = df.shape[0]
    total_features = df.shape[1]
    attrition_rate = df["Attrition"].mean() * 100
    avg_income = df["MonthlyIncome"].mean()

    col1.metric("Total Rows", total_rows)
    col2.metric("Total Features", total_features)
    col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")
    col4.metric("Average Income", f"${avg_income:,.0f}")

    st.markdown("---")

    # ----------------------------
    # TABS SECTION
    # ----------------------------
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Numerical Stats", "Categorical Stats"])

    with tab1:
        st.dataframe(df)

    with tab2:
        st.dataframe(df.describe())

    with tab3:
        cat_cols = df.select_dtypes(include="object")
        st.dataframe(cat_cols.describe())

# ======================================================
# PHASE 2 — UNIVARIATE ANALYSIS
# ======================================================
elif phase == "Phase 2: Univariate Analysis":

    st.title("📈 Univariate Analysis")

    feature = st.selectbox("Select Feature", df.columns)

    col1, col2 = st.columns(2)

    if df[feature].dtype == "object":

        # PIE CHART
        with col1:
            fig_pie = px.pie(df, names=feature, title=f"{feature} Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        # BAR CHART
        with col2:
            fig_bar = px.bar(df[feature].value_counts().reset_index(),
                             x="index", y=feature)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.info(f"Business Insight: Distribution of {feature} shows potential concentration risk.")

    else:

        # HISTOGRAM
        with col1:
            fig_hist = px.histogram(df, x=feature)
            st.plotly_chart(fig_hist, use_container_width=True)

        # BOXPLOT
        with col2:
            fig_box = px.box(df, y=feature)
            st.plotly_chart(fig_box, use_container_width=True)

        if df[feature].skew() > 1:
            st.warning("This feature is highly skewed. Consider transformation.")

# ======================================================
# PHASE 3 — BIVARIATE ANALYSIS
# ======================================================
elif phase == "Phase 3: Bivariate Analysis":

    st.title("🔍 Bivariate Analysis")

    relation_type = st.radio(
        "Select Relationship Type",
        ["Cat vs Cat", "Cat vs Num", "Num vs Num"]
    )

    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(exclude="object").columns

    if relation_type == "Cat vs Cat":

        col1, col2 = st.columns(2)
        cat1 = col1.selectbox("Select Categorical Feature 1", cat_cols)
        cat2 = col2.selectbox("Select Categorical Feature 2", cat_cols)

        fig = px.bar(df, x=cat1, color=cat2, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.warning("Potential high-risk segment identified in grouped categories.")

    elif relation_type == "Cat vs Num":

        col1, col2 = st.columns(2)
        cat = col1.selectbox("Select Categorical Feature", cat_cols)
        num = col2.selectbox("Select Numerical Feature", num_cols)

        fig = px.box(df, x=cat, y=num)
        st.plotly_chart(fig, use_container_width=True)

        st.info("Significant variation may indicate policy-level differences.")

    else:

        col1, col2 = st.columns(2)
        num1 = col1.selectbox("Select Numerical Feature 1", num_cols)
        num2 = col2.selectbox("Select Numerical Feature 2", num_cols)

        fig = px.scatter(df, x=num1, y=num2)
        st.plotly_chart(fig, use_container_width=True)

        st.info("Strong correlation may indicate hidden multicollinearity.")

# ======================================================
# PHASE 4 — MULTIVARIATE ANALYSIS
# ======================================================
elif phase == "Phase 4: Multivariate Analysis":

    st.title("📊 Correlation Heatmap")

    df_corr = df.copy()
    df_corr["Attrition"] = df_corr["Attrition"]

    corr = df_corr.select_dtypes(exclude="object").corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PHASE 5 — MACHINE LEARNING PREDICTOR
# ======================================================
elif phase == "Phase 5: Machine Learning Predictor":

    st.title("🤖 Attrition Prediction Engine")

    # Load model components
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    model_columns = pickle.load(open("model_columns.pkl", "rb"))

    st.subheader("Enter Employee Information")

    col1, col2 = st.columns(2)

    age = col1.slider("Age", 18, 60, 30)
    income = col1.number_input("Monthly Income", 1000, 20000, 5000)
    job_sat = col1.selectbox("Job Satisfaction", [1, 2, 3, 4])

    years = col2.slider("Years At Company", 0, 40, 5)
    overtime = col2.selectbox("OverTime", ["Yes", "No"])
    distance = col2.slider("Distance From Home", 1, 50, 10)
    env_sat = col2.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    wlb = col2.selectbox("Work Life Balance", [1, 2, 3, 4])

    if st.button("Predict"):

        input_dict = {
            "Age": age,
            "MonthlyIncome": income,
            "JobSatisfaction": job_sat,
            "YearsAtCompany": years,
            "OverTime": overtime,
            "DistanceFromHome": distance,
            "EnvironmentSatisfaction": env_sat,
            "WorkLifeBalance": wlb
        }

        input_df = pd.DataFrame([input_dict])

        input_df = pd.get_dummies(input_df)

        # Align columns safely
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error("⚠ High Risk of Attrition")
            st.progress(int(probability * 100))
            st.write(f"Probability: {probability*100:.2f}%")
            st.warning("Recommendation: Immediate engagement intervention required.")
        else:
            st.success("✅ Low Risk of Attrition")
            st.progress(int(probability * 100))
            st.write(f"Probability: {probability*100:.2f}%")
            st.info("Recommendation: Maintain current engagement strategy.")