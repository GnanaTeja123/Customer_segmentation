import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.run_pipeline import run_clustering_pipeline
from src.visualizer import (
    plot_pca_clusters,
    plot_elbow_curve,
    plot_silhouette,
    plot_dendrogram,
    plot_radar_chart,
)


# Page Settings
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation using Clustering")
st.markdown("Segment customers using various clustering models and visualize them interactively.")

# Sidebar: Select clustering model
model_type = st.sidebar.selectbox("Choose Clustering Algorithm", ['kmeans', 'agglomerative', 'gmm', 'dbscan'])

# Sidebar: Cluster slider (only for models with cluster control)
if model_type != 'dbscan':
    k = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)
else:
    k = None

# Run clustering pipeline
raw_df, rfm_clean, rfm_scaled, labels, silhouette, db_index = run_clustering_pipeline(k or 4, model_type)

# Display cluster metrics
st.sidebar.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette else "N/A")
st.sidebar.metric("Davies-Bouldin Index", f"{db_index:.3f}" if db_index else "N/A")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 PCA Cluster Plot",
    "📉 Elbow Curve",
    "🌐 Silhouette Plot",
    "🧱 Dendrogram",
    "🧭 Radar Chart"
])

with tab1:
    st.plotly_chart(plot_pca_clusters(rfm_scaled, labels), use_container_width=True)

with tab2:
    if model_type == 'kmeans':
        fig_elbow = plot_elbow_curve(rfm_scaled, max_k=10)
        st.pyplot(fig_elbow.gcf())
    else:
        st.info("Elbow Curve is only available for KMeans.")

with tab3:
    fig_sil = plot_silhouette(rfm_scaled, labels)
    st.pyplot(fig_sil.gcf())

with tab4:
    if model_type == 'agglomerative':
        fig_dendro = plot_dendrogram(rfm_scaled)
        st.pyplot(fig_dendro.gcf())
    else:
        st.info("Dendrogram is only available for Agglomerative Clustering.")

with tab5:
    fig_radar = plot_radar_chart(rfm_clean[['Recency', 'Frequency', 'Monetary']], labels)
    st.pyplot(fig_radar.gcf())

# Summary Table
st.subheader("📋 Cluster Summary (Mean RFM by Segment)")
summary = rfm_clean.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
st.dataframe(summary)

# Download segmented data
csv = rfm_clean.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Segmented Data", data=csv, file_name="customer_segments.csv", mime="text/csv")
