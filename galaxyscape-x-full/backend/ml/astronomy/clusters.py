"""
K-Means Clustering for Stellar Classification
==============================================

LEARNING: Unsupervised Learning - Clustering
---------------------------------------------

This module demonstrates K-Means clustering, an unsupervised ML algorithm that
groups similar data points together without labeled examples.

THEORY: What is Clustering?
---------------------------
Clustering finds natural groups (clusters) in data where:
- Points within a cluster are similar to each other
- Points in different clusters are dissimilar

Why is it "unsupervised"?
- No labels needed (unlike classification which needs "this is a red giant")
- Algorithm discovers patterns automatically
- Useful for exploratory data analysis

K-MEANS ALGORITHM:
------------------
K-Means is an iterative algorithm that partitions data into K clusters:

1. INITIALIZATION: Randomly place K cluster centers (centroids)
2. ASSIGNMENT: Assign each point to nearest centroid
3. UPDATE: Move centroids to center of their assigned points
4. REPEAT: Steps 2-3 until convergence (centroids stop moving)

MATHEMATICAL FOUNDATION:
-----------------------
K-Means minimizes within-cluster sum of squares (WCSS):

    WCSS = Σᵢ Σⱼ ||xᵢⱼ - μᵢ||²

Where:
    - xᵢⱼ = j-th point in cluster i
    - μᵢ = centroid (mean) of cluster i
    - ||·|| = Euclidean distance

The algorithm finds K centroids that minimize total distance from points to centroids.

WHY K CLUSTERS?
--------------
K must be chosen beforehand. Common methods:
- Elbow method: Plot WCSS vs K, find "elbow" where improvement slows
- Domain knowledge: Know you want 5 stellar types? Use K=5
- Silhouette score: Measure cluster quality, choose K with best score

DIMENSIONALITY REDUCTION (PCA):
-------------------------------
High-dimensional data (many features) is hard to visualize and cluster.

PCA (Principal Component Analysis) reduces dimensions:
- Finds directions of maximum variance (principal components)
- Projects data onto 2D plane for visualization
- Preserves most information in fewer dimensions

Formula: X_2d = X × W
Where W = matrix of principal components (eigenvectors)

FEATURE SCALING (StandardScaler):
---------------------------------
Different features have different scales:
- Temperature: 3000-6000 K
- Mass: 0.5-2.0 solar masses
- Luminosity: 0.1-10 solar luminosities

Without scaling, large-scale features dominate distance calculations.

StandardScaler normalizes: z = (x - μ) / σ
- Mean (μ) = 0
- Standard deviation (σ) = 1
- All features on same scale

LEARNING CHECKPOINT:
-------------------
1. Why use PCA before clustering?
   → Reduces dimensions for visualization and faster computation

2. What does K=5 mean?
   → Algorithm will find 5 distinct groups in the data

3. Why standardize features?
   → Ensures all features contribute equally to distance calculations
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ml.data_loaders import load_astronomy_data


def get_cluster_assignments(n_clusters=5, df=None):
    """
    Perform K-Means clustering on stellar features.
    
    LEARNING: This function demonstrates the complete clustering pipeline:
    1. Data Loading
    2. Feature Selection (numeric columns only)
    3. Feature Scaling (StandardScaler)
    4. Dimensionality Reduction (PCA to 2D)
    5. Clustering (K-Means)
    6. Visualization Preparation (2D points, labels, centers)
    
    Args:
        n_clusters: Number of clusters to find (K parameter)
        df: Optional DataFrame to use (if None, loads from data_loaders)
    
    Returns:
        dict with points (2D coordinates), cluster_labels, cluster_centers, n_clusters
    """
    # STEP 1: DATA LOADING
    # ---------------------
    if df is None:
        df = load_astronomy_data()
    if df is None or len(df) == 0:
        return {
            'points': [],
            'cluster_labels': [],
            'cluster_centers': [],
            'n_clusters': 0
        }

    # STEP 2: FEATURE SELECTION
    # --------------------------
    # LEARNING: Clustering works on numeric features only
    # Select all numeric columns (temperature, mass, radius, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # LEARNING: Remove target variables and existing cluster labels
    # We don't want to cluster on things we're trying to predict!
    for col in ['age', 'stellar_age', 'cluster', 'cluster_label']:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # STEP 3: HANDLE INSUFFICIENT FEATURES
    # -------------------------------------
    # LEARNING: Need at least 2 features for meaningful clustering
    if len(numeric_cols) < 2:
        # Generate synthetic features if data is insufficient
        df['feat1'] = np.random.randn(len(df))
        df['feat2'] = np.random.randn(len(df))
        numeric_cols = ['feat1', 'feat2']

    # STEP 4: PREPARE FEATURE MATRIX
    # --------------------------------
    # LEARNING: X is the feature matrix (n_samples × n_features)
    # Select top 10 features to avoid curse of dimensionality
    X = df[numeric_cols[:10]].fillna(0)  # Fill missing values with 0
    
    # STEP 5: FEATURE SCALING
    # -----------------------
    # LEARNING: StandardScaler normalizes features to mean=0, std=1
    # Why? Different features have different scales:
    #   - Temperature: 3000-6000 K (large numbers)
    #   - Mass: 0.5-2.0 (small numbers)
    # Without scaling, temperature would dominate distance calculations!
    #
    # Formula: z = (x - μ) / σ
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # X_scaled now has mean=0, std=1 for each feature

    # STEP 6: DIMENSIONALITY REDUCTION (PCA)
    # ---------------------------------------
    # LEARNING: PCA reduces high-dimensional data to 2D for visualization
    #
    # Why 2D?
    # - Can visualize on screen (2D plot)
    # - Easier to understand cluster structure
    # - Faster computation
    #
    # How PCA works:
    # 1. Finds directions of maximum variance (principal components)
    # 2. Projects data onto these directions
    # 3. Preserves most information in fewer dimensions
    #
    # Mathematical: X_2d = X_scaled × W
    # Where W = matrix of principal components (eigenvectors of covariance matrix)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    # X_2d shape: (n_samples, 2) - ready for 2D visualization

    # STEP 7: VALIDATE CLUSTER COUNT
    # -------------------------------
    # LEARNING: Can't have more clusters than data points!
    # Rule of thumb: Need at least 2 points per cluster
    n_clusters = min(n_clusters, len(X_2d) // 2) if len(X_2d) > 2 else 2
    if n_clusters < 2:
        n_clusters = 2  # Minimum 2 clusters

    # STEP 8: K-MEANS CLUSTERING
    # ---------------------------
    # LEARNING: K-Means algorithm partitions data into K groups
    #
    # Algorithm steps:
    # 1. Initialize K centroids (cluster centers) randomly
    # 2. Assign each point to nearest centroid
    # 3. Update centroids to center of assigned points
    # 4. Repeat steps 2-3 until convergence
    #
    # Parameters:
    # - n_clusters: Number of clusters (K)
    # - random_state: Seed for reproducibility
    # - n_init: Number of times to run with different initializations (best result kept)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # LEARNING: fit_predict does both training and prediction
    # - fit: Learn cluster centers from data
    # - predict: Assign each point to a cluster
    labels = kmeans.fit_predict(X_2d)
    
    # LEARNING: Cluster centers are the centroids (mean of points in each cluster)
    centers_2d = kmeans.cluster_centers_
    # Shape: (n_clusters, 2) - one 2D point per cluster

    # STEP 9: PREPARE RESULTS FOR VISUALIZATION
    # -----------------------------------------
    # LEARNING: Return format optimized for frontend visualization
    # - points: 2D coordinates for scatter plot
    # - cluster_labels: Which cluster each point belongs to (for coloring)
    # - cluster_centers: Centroids to display as markers
    return {
        'points': [[float(x), float(y)] for x, y in X_2d],
        'cluster_labels': labels.tolist(),
        'cluster_centers': [[float(x), float(y)] for x, y in centers_2d],
        'n_clusters': int(n_clusters)
    }
