from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def scale_data(rfm):
    scaler = StandardScaler()
    return scaler.fit_transform(rfm), scaler

def apply_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(data), model

def apply_dbscan(data):
    model = DBSCAN(eps=1, min_samples=5)
    return model.fit_predict(data), model

def apply_gmm(data, n_clusters):
    model = GaussianMixture(n_components=n_clusters, random_state=42)
    return model.fit_predict(data), model

def apply_agglomerative(data, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(data), model