import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import numpy as np
import plotly.express as px


def plot_pca_clusters(data_scaled, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    df_plot = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    df_plot["Cluster"] = labels.astype(str)
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster", title="PCA Cluster Visualization")
    return fig


def plot_elbow_curve(data_scaled, max_k=10):
    from sklearn.cluster import KMeans
    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        distortions.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, distortions, marker='o')
    ax.set_title('Elbow Curve for Optimal K')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    return plt


def plot_silhouette(data_scaled, labels):
    silhouette_vals = silhouette_samples(data_scaled, labels)
    silhouette_avg = silhouette_score(data_scaled, labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_lower = 10
    for i in np.unique(labels):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot for Clusters")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    return plt


def plot_dendrogram(data_scaled):
    linked = linkage(data_scaled, 'ward')
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, ax=ax)
    ax.set_title('Hierarchical Clustering Dendrogram')
    return plt


def plot_radar_chart(rfm_df, labels):
    df_radar = rfm_df.copy()
    df_radar['Cluster'] = labels
    df_mean = df_radar.groupby('Cluster').mean()
    categories = ['Recency', 'Frequency', 'Monetary']
    num_clusters = len(df_mean)

    # Normalize values to [0, 1] for radar
    df_norm = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min())

    # Setup plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # loop back

    for i in range(num_clusters):
        values = df_norm.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_title('Radar Chart of Cluster Averages', size=14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    return plt
