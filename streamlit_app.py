# app.py
# Customer Profiling + Segmentation + Campaign Response App
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Customer Profiling & Targeting",
    layout="wide"
)

# Features used in clustering & prediction
CLUSTER_VARS = [
    "Income",
    "NumCatalogPurchases",
    "Dependency_Ratio",
    "MntFruits",
    "MntWines",
    "MntMeatProducts",
    "NumWebVisitsMonth",
    "NumStorePurchases",
]

# ORIGINAL PERSONA MEANINGS (as you requested)
PERSONA_NAMES = {
    0: "Value-Conscious Digital Browsers",
    1: "Store-Buying Families",
    2: "Affluent Premium Spenders",
    3: "Luxury Impulse Shoppers",
}


# -----------------------------------------------------------
# 2. UTILITY FUNCTIONS
# -----------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering used in the project."""

    # Total expenditure
    spend_cols = [
        "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    ]
    df["Total_Expenditure"] = df[spend_cols].sum(axis=1)

    # Tenure
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    current_date = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (current_date - df["Dt_Customer"]).dt.days
    df["Customer_Tenure_Months"] = df["Customer_Tenure_Days"] / 30

    # Dependency ratio (assumed 2 adults per household)
    df["Dependency_Ratio"] = (df["Kidhome"] + df["Teenhome"]) / 2

    # Engagement score
    df["Engagement_Score"] = df["NumWebVisitsMonth"] * 0.4 + df["NumStorePurchases"] * 0.6

    # Campaign response target
    cmp_cols = [
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
        "AcceptedCmp4", "AcceptedCmp5", "Response",
    ]
    df["Campaign_Response"] = (df[cmp_cols].sum(axis=1) > 0).astype(int)

    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train scaler, PCA, KMeans, Logistic Regression."""

    df = feature_engineering(df)
    df_model = df.dropna(subset=CLUSTER_VARS + ["Campaign_Response"]).copy()

    X = df_model[CLUSTER_VARS].values
    y = df_model["Campaign_Response"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    df_model["Cluster"] = clusters

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_train, y_train)
    acc = logit.score(X_test, y_test)

    return df_model, scaler, pca, kmeans, logit, acc


def predict_single_customer(input_dict, scaler, pca, kmeans, logit):
    x_df = pd.DataFrame([input_dict])[CLUSTER_VARS]
    x_scaled = scaler.transform(x_df)
    x_pca = pca.transform(x_scaled)

    cluster_id = int(kmeans.predict(x_pca)[0])
    persona = PERSONA_NAMES.get(cluster_id, f"Cluster {cluster_id}")

    prob = float(logit.predict_proba(x_scaled)[0, 1])

    return cluster_id, persona, prob


# -----------------------------------------------------------
# 3. MAIN APP UI
# -----------------------------------------------------------

st.title("Customer Profiling & Targeted Marketing App")

st.markdown("""
This app demonstrates the complete **Customer Segmentation + Campaign Response**
pipeline using Machine Learning:

✔ Feature Engineering  
✔ PCA Dimensionality Reduction  
✔ K-Means Segmentation  
✔ Logistic Regression for Campaign Response  
""")

# Sidebar Inputs
st.sidebar.header("📌 App Settings")
data_path = st.sidebar.text_input("Dataset Path", "Customer_Profiling.csv")

with st.spinner("Training models... please wait ⏳"):
    df_raw = load_data(data_path)
    df_model, scaler, pca, kmeans, logit, test_acc = train_models(df_raw)

st.sidebar.success("Models trained successfully!")
st.sidebar.write(f"Logistic Regression Accuracy: **{test_acc:.2%}**")

mode = st.sidebar.radio(
    "Choose a Feature",
    ["Single Customer Input", "Upload & Score File", "View Cluster Profiles"]
)

# MODE 1: SINGLE INPUT
if mode == "Single Customer Input":
    st.subheader("Predict Segment & Response for One Customer")

    col1, col2, col3 = st.columns(3)

    with col1:
        Income = st.number_input("Income", 0, 500000, 30000)
        NumCatalogPurchases = st.number_input("Catalog Purchases", 0, 50, 2)
        Dependency_Ratio = st.number_input("Dependency Ratio", 0.0, 5.0, 0.5, 0.1)

    with col2:
        MntFruits = st.number_input("Spend on Fruits", 0, 5000, 50)
        MntWines = st.number_input("Spend on Wines", 0, 5000, 100)
        MntMeatProducts = st.number_input("Spend on Meat Products", 0, 5000, 100)

    with col3:
        NumWebVisitsMonth = st.number_input("Web Visits / Month", 0, 30, 5)
        NumStorePurchases = st.number_input("Store Purchases", 0, 50, 5)

    if st.button("Predict"):
        customer_data = locals()
        cluster_id, persona, prob = predict_single_customer(
            customer_data, scaler, pca, kmeans, logit
        )
        st.success(f"Customer Segment: **{cluster_id} - {persona}**")
        st.info(f"Predicted Campaign Response Probability: **{prob:.1%}**")

# MODE 2: BATCH SCORING
elif mode == "Upload & Score File":
    st.subheader("Score Multiple Customers")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_new = pd.read_csv(uploaded)
        missing = [c for c in CLUSTER_VARS if c not in df_new.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            X_scaled_new = scaler.transform(df_new[CLUSTER_VARS])
            X_pca_new = pca.transform(X_scaled_new)

            df_new["Cluster"] = kmeans.predict(X_pca_new)
            df_new["Persona"] = df_new["Cluster"].map(PERSONA_NAMES)
            df_new["Response_Probability"] = logit.predict_proba(X_scaled_new)[:, 1]

            st.dataframe(df_new.head())
            st.download_button(
                "Download Results",
                df_new.to_csv(index=False),
                file_name="customer_scored.csv",
                mime="text/csv",
            )

# MODE 3: CLUSTER PROFILES
elif mode == "View Cluster Profiles":
    st.subheader("Customer Segment Profiles")

    profile = df_model.groupby("Cluster")[CLUSTER_VARS].mean().round(2)
    profile["Customers"] = df_model["Cluster"].value_counts().sort_index()
    profile["Persona"] = [PERSONA_NAMES[i] for i in profile.index]

    st.dataframe(profile)

    st.markdown("📌 Use these profiles for targeted marketing strategies.")

