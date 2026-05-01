"""
K-means clustering prediction utilities.
Usage in web app: from models.kmeans_predict import predict_cluster
"""

import joblib
import numpy as np
import os
from sklearn.decomposition import PCA

# Base 11 features used for training kmeans
KMEANS_BASE_FEATURES = [
    "Mean",
    "Variance",
    "Skewness",
    "Kurtosis",
    "Power_Delta",
    "Power_Theta",
    "Power_Alpha",
    "Power_Beta",
    "Theta_Alpha_Ratio",
    "Spectral_Entropy",
    "Rolling_Var",
]


def get_kmeans_models(models_dir='.'):
    """Load kmeans scaler and model. PCA pickles are optional and not required."""
    scaler = joblib.load(os.path.join(models_dir, 'kmeans_scaler.pkl'))
    kmeans = joblib.load(os.path.join(models_dir, 'kmeans_model.pkl'))
    return scaler, kmeans

def predict_cluster(X_new, models_dir='.'):
    """
    Predict cluster labels for new data.
    
    Args:
        X_new: array-like, shape (n_samples, n_features)
               If 19 features, will extract only base 11
        models_dir: path to models directory
    
    Returns:
        cluster_labels: array, shape (n_samples,)
        distances: array, shape (n_samples, n_clusters)
        cluster_stats: dict with cluster statistics
    """
    scaler, kmeans = get_kmeans_models(models_dir)

    expected_features = int(getattr(scaler, 'n_features_in_', X_new.shape[1]))
    if X_new.shape[1] < expected_features:
        raise ValueError(
            f"Input has {X_new.shape[1]} features, but kmeans scaler expects {expected_features}."
        )
    if X_new.shape[1] > expected_features:
        X_new = X_new[:, :expected_features]
    
    X_new_scaled = scaler.transform(X_new)
    cluster_labels = kmeans.predict(X_new_scaled)
    
    # Calculate distances to cluster centers
    distances = kmeans.transform(X_new_scaled)  # shape (n_samples, n_clusters)
    
    # Cluster statistics
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = {
        'clusters': unique_clusters.tolist(),
        'counts': counts.tolist(),
        'percentages': (100 * counts / len(cluster_labels)).tolist()
    }
    
    return cluster_labels, distances, cluster_stats

def predict_cluster_with_pca(X_new, models_dir='.', n_components=2):
    """
    Predict cluster and get PCA visualization coordinates.
    
    Args:
        X_new: array-like, shape (n_samples, n_features)
               If 19 features, will extract only base 11
        models_dir: path to models directory
        n_components: 2 or 3 for visualization
    
    Returns:
        cluster_labels: array, shape (n_samples,)
        pca_coords: array, shape (n_samples, n_components)
        cluster_stats: dict with cluster info
    """
    scaler, kmeans = get_kmeans_models(models_dir)

    expected_features = int(getattr(scaler, 'n_features_in_', X_new.shape[1]))
    if X_new.shape[1] < expected_features:
        raise ValueError(
            f"Input has {X_new.shape[1]} features, but kmeans scaler expects {expected_features}."
        )
    if X_new.shape[1] > expected_features:
        X_new = X_new[:, :expected_features]
    
    X_new_scaled = scaler.transform(X_new)
    cluster_labels = kmeans.predict(X_new_scaled)
    
    # Compute PCA on-the-fly for visualization (do not require saved PCA objects)
    pca = PCA(n_components=2 if n_components == 2 else 3)
    pca_coords = pca.fit_transform(X_new_scaled)
    
    # Cluster statistics
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_stats = {
        'clusters': unique_clusters.tolist(),
        'counts': counts.tolist(),
        'percentages': (100 * counts / len(cluster_labels)).tolist()
    }
    
    return cluster_labels, pca_coords, cluster_stats

if __name__ == '__main__':
    # Test
    print("K-means predict utilities loaded successfully.")
