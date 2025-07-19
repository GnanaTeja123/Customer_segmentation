# 🧠 Customer Segmentation using Clustering

This interactive Streamlit app segments customers based on RFM (Recency, Frequency, Monetary) analysis using various clustering algorithms. It helps businesses understand customer behavior and optimize marketing strategies by grouping similar customers together.

---

## 📌 Features

- 📈 Apply clustering algorithms: **KMeans**, **Agglomerative**, **Gaussian Mixture**, **DBSCAN**
- 🎯 Adjustable number of clusters (where applicable)
- 📊 Visualizations:
  - PCA Cluster Plot
  - Elbow Curve
  - Silhouette Plot
  - Dendrogram
  - Radar Chart
- 📥 Download the final segmented customer dataset
- 📋 View summary statistics for each cluster

---

## 📁 Project Structure

customer_segmentation/
├── requirements.txt
├── run_pipeline.py
├── app/
│   └── streamlit_app.py
├── data/raw/
│   └── Online Retail.xlsx
└── src/
    ├── clustering_models.py
    ├── data_loader.py
    ├── evaluator.py
    ├── rfm.py
    └── visualizer.py

---

## Requirements
All dependencies are listed in requirements.txt. Main libraries include:
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
scipy

---

## Concepts Used
RFM (Recency, Frequency, Monetary) Analysis
Data Cleaning & Normalization
Dimensionality Reduction using PCA
Clustering Algorithms:
KMeans
Agglomerative Clustering
Gaussian Mixture Models (GMM)
DBSCAN
Clustering Evaluation Metrics:
Silhouette Score
Davies Bouldin Index
Data Visualization with Plotly, Matplotlib, Seaborn

---

## Sample Output
Sidebar shows Silhouette Score and Davies-Bouldin Index
Clustered customers displayed using interactive visualizations
Segmented dataset available for download
Mean RFM values for each cluster shown in a summary table

## Contact
For questions, feedback, or collaborations:
Gnana Teja K
Email: [gnanateja9898@gmail.com]
GitHub: https://github.com/GnanaTeja123
