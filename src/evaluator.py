
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(data, labels):
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    db_score = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else -1
    return silhouette, db_score