import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load Dataset
def load_data():
    df = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    )
    df.dropna(inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

# RFM Features
def generate_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# Clustering Pipeline
def run_clustering_pipeline(k=4, model_type='kmeans'):
    df = load_data()
    rfm = generate_rfm(df)
    rfm_clean = rfm[rfm['Monetary'] > 0].copy()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_clean[['Recency', 'Frequency', 'Monetary']])

    if model_type == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(rfm_scaled)
    elif model_type == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(rfm_scaled)
    elif model_type == 'gmm':
        model = GaussianMixture(n_components=k, random_state=42)
        labels = model.fit_predict(rfm_scaled)
    elif model_type == 'dbscan':
        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(rfm_scaled)
    else:
        raise ValueError("Invalid model type selected.")

    rfm_clean['Cluster'] = labels

    # Metrics (excluding noise in DBSCAN)
    valid_labels = labels[labels != -1] if -1 in labels else labels
    valid_scaled = rfm_scaled[labels != -1] if -1 in labels else rfm_scaled

    silhouette = silhouette_score(valid_scaled, valid_labels) if len(set(valid_labels)) > 1 else None
    db_index = davies_bouldin_score(valid_scaled, valid_labels) if len(set(valid_labels)) > 1 else None

    return df, rfm_clean, rfm_scaled, labels, silhouette, db_index
