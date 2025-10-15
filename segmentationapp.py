import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------
# 1️⃣ PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# ---------------------------------------------------------------
# 2️⃣ SIMPLE LOGIN PAGE (NO LOGOUT / NO RERUN)
# ---------------------------------------------------------------
users = {"admin": "1234", "user": "abcd"}  # username: password

# Initialize login state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# If not authenticated, show login form
if not st.session_state.authenticated:
    st.title("🔐 Customer Segmentation Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.authenticated = True
            st.success("✅ Login successful! Redirecting to dashboard...")
        else:
            st.error("❌ Invalid username or password.")

# ---------------------------------------------------------------
# 3️⃣ MAIN APP (AFTER LOGIN)
# ---------------------------------------------------------------
if st.session_state.authenticated:
    st.title("🧠 Customer Segmentation Dashboard")

    # -----------------------------
    # 1. Upload Dataset
    # -----------------------------
    st.header("📂 Upload Customer Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------
        # 2. Data Understanding
        # -----------------------------
        st.header("🧩 Data Understanding")
        st.write("Shape of Data:", df.shape)
        st.write("Missing Values per Column:")
        st.write(df.isnull().sum())

        # Clean categorical data
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

        # Remove ID-like columns
        for col in df.columns:
            if "ID" in col or "Id" in col:
                df.drop(col, axis=1, inplace=True)

        # Numeric data only
        X = df.select_dtypes(include=["float64", "int64"])

        # -----------------------------
        # 3. Feature Scaling
        # -----------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------
        # 4. KMeans Clustering
        # -----------------------------
        st.header("⚙️ Clustering Configuration")
        k = st.slider("Select number of clusters (K):", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["Cluster"] = kmeans.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, df["Cluster"])
        st.metric("Silhouette Score", f"{silhouette:.3f}")

        # -----------------------------
        # 5. PCA Dimensionality Reduction
        # -----------------------------
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df["PCA1"] = pca_result[:, 0]
        df["PCA2"] = pca_result[:, 1]

        # -----------------------------
        # 6. Visualization Tabs
        # -----------------------------
        st.header("📊 Advanced Visualizations")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 PCA Scatter",
            "📉 3D Scatter",
            "🔥 Heatmap",
            "🎯 Radar Chart",
            "📦 Box Plot",
            "📊 Pair Plot"
        ])

        # PCA 2D Scatter
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=80)
            plt.title("Clusters in 2D PCA Space")
            st.pyplot(fig)

        # 3D Scatter
        with tab2:
            if X.shape[1] >= 3:
                pca_3d = PCA(n_components=3)
                X_pca3 = pca_3d.fit_transform(X_scaled)
                fig3d = px.scatter_3d(
                    x=X_pca3[:, 0],
                    y=X_pca3[:, 1],
                    z=X_pca3[:, 2],
                    color=df["Cluster"].astype(str),
                    title="3D Cluster Visualization",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.warning("Need at least 3 features for 3D visualization.")

        # Heatmap
        with tab3:
            st.subheader("Feature Means per Cluster (Heatmap)")
            cluster_means = df.groupby("Cluster")[X.columns].mean()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.heatmap(cluster_means, annot=True, cmap="coolwarm")
            st.pyplot(fig2)

        # Radar Chart
        with tab4:
            st.subheader("Radar Chart - Cluster Profiles")
            cluster_profile = df.groupby("Cluster")[X.columns].mean()
            categories = list(cluster_profile.columns)
            fig_radar = go.Figure()
            for i in range(k):
                fig_radar.add_trace(go.Scatterpolar(
                    r=cluster_profile.iloc[i].values,
                    theta=categories,
                    fill='toself',
                    name=f'Cluster {i}'
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Box Plot
        with tab5:
            st.subheader("Distribution of Features per Cluster")
            selected_feature = st.selectbox("Select feature for box plot:", X.columns)
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            sns.boxplot(x="Cluster", y=selected_feature, data=df, palette="Set2")
            st.pyplot(fig5)

        # Pair Plot
        with tab6:
            st.subheader("Pairplot of Features by Cluster")
            fig6 = sns.pairplot(df, vars=X.columns, hue="Cluster", palette="husl")
            st.pyplot(fig6)

        # -----------------------------
        # 7. Cluster Summary + Download
        # -----------------------------
        st.header("🧾 Cluster Summary")
        cluster_summary = df.groupby("Cluster")[X.columns].mean().round(2)
        st.dataframe(cluster_summary)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Clustered Data", data=csv, file_name="customer_segments.csv")

    else:
        st.info("⬆️ Please upload a CSV file to begin segmentation.")

