import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D


BASE_FEATURE_COLS = [
    'Mean',
    'Variance',
    'Skewness',
    'Kurtosis',
    'Power_Delta',
    'Power_Theta',
    'Power_Alpha',
    'Power_Beta',
    'Theta_Alpha_Ratio',
    'Spectral_Entropy',
    'Rolling_Var',
]

DERIVED_FEATURE_COLS = [
    'Delta_diff',
    'Theta_diff',
    'Alpha_diff',
    'Beta_diff',
    'Delta_Theta_ratio',
    'Alpha_Beta_ratio',
    'Rolling_Mean_Delta',
    'Rolling_Mean_Theta',
]


def add_derived_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to convert base-11 feature table into 19 features."""
    df_out = df_in.copy()
    missing = [c for c in BASE_FEATURE_COLS if c not in df_out.columns]
    if missing:
        raise ValueError(f"Missing base features in CSV: {missing}")

    df_out['Delta_diff'] = df_out['Power_Delta'].diff().fillna(0)
    df_out['Theta_diff'] = df_out['Power_Theta'].diff().fillna(0)
    df_out['Alpha_diff'] = df_out['Power_Alpha'].diff().fillna(0)
    df_out['Beta_diff'] = df_out['Power_Beta'].diff().fillna(0)
    df_out['Delta_Theta_ratio'] = df_out['Power_Delta'] / (df_out['Power_Theta'] + 1e-6)
    df_out['Alpha_Beta_ratio'] = df_out['Power_Alpha'] / (df_out['Power_Beta'] + 1e-6)
    df_out['Rolling_Mean_Delta'] = df_out['Power_Delta'].rolling(5, min_periods=1).mean()
    df_out['Rolling_Mean_Theta'] = df_out['Power_Theta'].rolling(5, min_periods=1).mean()

    return df_out

df = pd.read_csv('all_subjects_features.csv')
print(f"Clustering train set: {df.shape[0]} epochs, {df.shape[1]} columns (includes Label)")

# Keep label for export if available, but cluster only on features.
label_series = df['Label'] if 'Label' in df.columns else None
df_features = df.drop(columns=['Label'], errors='ignore')

# Synchronize to 19 features if CSV currently has only base 11.
has_all_derived = all(col in df_features.columns for col in DERIVED_FEATURE_COLS)
if not has_all_derived:
    print("Derived features not found in CSV. Generating 8 derived features to reach 19...")
    df_features = add_derived_features(df_features)

final_feature_cols = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS
X_df = df_features[final_feature_cols].copy()

# Save synchronized 19-feature CSV for reuse.
df_19 = X_df.copy()
if label_series is not None:
    df_19['Label'] = label_series.values
df_19.to_csv('all_subjects_features_19.csv', index=False)
print(f"Saved synchronized feature table: all_subjects_features_19.csv ({df_19.shape[1]} columns)")

X = X_df.values
print(f"Features used for clustering: {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data stats: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

print("\nFinding optimal number of clusters (Elbow method)...")
inertias = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=5)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    print(f"k={k}: inertia={km.inertia_:.2f}")

# Tìm k tối ưu bằng elbow (kiểm tra độ suy giảm)
diffs = np.diff(inertias)
second_diffs = np.diff(diffs)
optimal_k = list(K_range)[1:][np.argmin(second_diffs)] if len(second_diffs) > 0 else 3
print(f"\nOptimal k: {optimal_k} (from Elbow method)")

print(f"\nTraining K-means with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels

sil_score = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")
print(f"Inertia: {kmeans.inertia_:.2f}")

print("\nCluster statistics:")
for cluster_id in range(optimal_k):
    count = (cluster_labels == cluster_id).sum()
    pct = 100 * count / len(cluster_labels)
    print(f"Cluster {cluster_id}: {count} epochs ({pct:.1f}%)")

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
print(f"\nPCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.2%}")

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)
print(f"PCA 3D explained variance: {pca_3d.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
            c=df['Cluster'], cmap='viridis', s=5)
plt.title(f'K-means Clusters (PCA 2D, k={optimal_k})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')

plt.subplot(1, 2, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
            c=range(len(X_pca_2d)), cmap='tab10', s=5, alpha=0.6)
plt.title('Data Distribution (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.tight_layout()
plt.savefig('kmeans_pca_2d.png', dpi=100)
plt.close()

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
            c=df['Cluster'], cmap='viridis', s=5)
ax1.set_title(f'K-means Clusters (PCA 3D, k={optimal_k})')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')
ax1.set_zlabel('PCA 3')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
            c=range(len(X_pca_3d)), cmap='tab10', s=5, alpha=0.6)
ax2.set_title('Data Distribution (PCA 3D)')
ax2.set_xlabel('PCA 1')
ax2.set_ylabel('PCA 2')
ax2.set_zlabel('PCA 3')

plt.tight_layout()
plt.savefig('kmeans_pca_3d.png', dpi=100)
plt.close()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
# Second derivative - elbow detection
second_diffs = np.diff(np.diff(inertias))
k_range_2 = list(K_range)[2:]
plt.plot(k_range_2, second_diffs, 'go-')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Second Derivative')
plt.title('Elbow Detection')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('kmeans_optimization.png', dpi=100)
plt.close()

print("\nSaved plots: kmeans_pca_2d.png, kmeans_pca_3d.png, kmeans_optimization.png")

models_dir = '../models'
os.makedirs(models_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(models_dir, 'kmeans_scaler.pkl'))
joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.pkl'))

print(f"\nSaved models:")
print(f"  - {models_dir}/kmeans_scaler.pkl")
print(f"  - {models_dir}/kmeans_model.pkl")
 

print("\nClustering artifacts are ready for web integration.")
print("\nClustering completed.")