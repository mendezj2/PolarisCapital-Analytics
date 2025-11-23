"""
Anomaly Detection using Isolation Forest
=========================================

LEARNING: Unsupervised Anomaly Detection
-----------------------------------------

This module demonstrates Isolation Forest, an algorithm that identifies outliers
and anomalies in data without labeled examples.

THEORY: What are Anomalies?
---------------------------
Anomalies (outliers) are data points that:
- Deviate significantly from normal patterns
- Are rare or unusual
- May indicate errors, fraud, or interesting discoveries

Examples in astronomy:
- Stars with unusual temperature-mass relationships
- Extremely fast rotators
- Stars with anomalous chemical compositions

ISOLATION FOREST ALGORITHM:
---------------------------
Isolation Forest is based on a simple idea: anomalies are easier to isolate.

Key Concept:
- Normal points: Dense, hard to isolate (need many splits)
- Anomalies: Sparse, easy to isolate (few splits needed)

How it works:
1. Build random binary trees (like decision trees)
2. For each point, count path length (how many splits to isolate it)
3. Short path = anomaly (easy to isolate)
4. Long path = normal (hard to isolate)

MATHEMATICAL FOUNDATION:
-----------------------
Anomaly Score:
    s(x, n) = 2^(-E(h(x)) / c(n))

Where:
    - h(x) = path length (number of edges from root to leaf)
    - E(h(x)) = expected path length
    - c(n) = normalization constant (average path length for n samples)

Interpretation:
    - s ≈ 1: Anomaly (short path, easy to isolate)
    - s ≈ 0: Normal (long path, hard to isolate)
    - s < 0.5: Typically considered normal
    - s > 0.5: Typically considered anomaly

CONTAMINATION PARAMETER:
-----------------------
contamination = expected proportion of anomalies in data

Examples:
    - contamination=0.1: Expect 10% of data to be anomalies
    - contamination=0.05: Expect 5% (more conservative)
    - contamination=0.2: Expect 20% (more aggressive)

The algorithm labels the top contamination% of points as anomalies.

WHY ISOLATION FOREST?
---------------------
Advantages:
1. Unsupervised: No labeled examples needed
2. Fast: O(n log n) complexity
3. Handles high dimensions well
4. Works with mixed data types

Alternative methods:
- Local Outlier Factor (LOF): Density-based, slower
- One-Class SVM: Requires careful parameter tuning
- Statistical methods (Z-score): Assumes normal distribution

DIMENSIONALITY REDUCTION (PCA):
-------------------------------
Before anomaly detection, we reduce dimensions:
- High-dimensional data: Hard to visualize and detect anomalies
- 2D projection: Easier to understand and visualize
- PCA preserves most variance while reducing dimensions

LEARNING CHECKPOINT:
-------------------
1. Why are anomalies "easy to isolate"?
   → They're far from normal points, so few splits needed to separate them

2. What does contamination=0.1 mean?
   → Algorithm expects 10% of data to be anomalies

3. Why use PCA before anomaly detection?
   → Reduces dimensions for better performance and visualization
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ml.data_loaders import load_astronomy_data


def get_anomaly_scores(contamination=0.1):
    """
    Detect anomalies using Isolation Forest algorithm.
    
    LEARNING: This function demonstrates:
    1. Feature preparation (numeric columns, scaling)
    2. Dimensionality reduction (PCA)
    3. Anomaly detection (Isolation Forest)
    4. Score interpretation (anomaly scores and binary flags)
    
    Args:
        contamination: Expected proportion of anomalies (0.0 to 0.5)
                       - 0.1 = expect 10% of data to be anomalies
                       - Lower = more conservative (fewer anomalies flagged)
                       - Higher = more aggressive (more anomalies flagged)
    
    Returns:
        dict with points (2D), anomaly_scores, is_anomaly flags, n_anomalies count
    """
    # STEP 1: DATA LOADING
    # ---------------------
    df = load_astronomy_data()
    if df is None or len(df) == 0:
        return {
            'points': [],
            'anomaly_scores': [],
            'is_anomaly': [],
            'n_anomalies': 0
        }

    # STEP 2: FEATURE SELECTION
    # -------------------------
    # LEARNING: Anomaly detection works on numeric features
    # Select all numeric columns (temperature, mass, radius, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # LEARNING: Remove target variables (we're not detecting anomalies in predictions!)
    for col in ['age', 'stellar_age', 'cluster']:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # STEP 3: HANDLE INSUFFICIENT FEATURES
    # ------------------------------------
    if len(numeric_cols) < 2:
        # Generate synthetic features if needed
        df['feat1'] = np.random.randn(len(df))
        df['feat2'] = np.random.randn(len(df))
        numeric_cols = ['feat1', 'feat2']

    # STEP 4: PREPARE FEATURE MATRIX
    # --------------------------------
    # LEARNING: X is the feature matrix (n_samples × n_features)
    # Select top 10 features to balance information and performance
    X = df[numeric_cols[:10]].fillna(0)  # Fill missing values

    # STEP 5: FEATURE SCALING
    # -----------------------
    # LEARNING: StandardScaler normalizes features to mean=0, std=1
    # Critical for anomaly detection: features on different scales can skew results
    #
    # Example problem without scaling:
    #   - Temperature: 3000-6000 (large values)
    #   - Mass: 0.5-2.0 (small values)
    #   - Temperature would dominate distance calculations!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 6: DIMENSIONALITY REDUCTION (PCA)
    # ---------------------------------------
    # LEARNING: Reduce to 2D for visualization and faster computation
    # PCA finds directions of maximum variance and projects data onto them
    #
    # Why 2D?
    # - Visualization: Can plot on screen
    # - Performance: Faster anomaly detection
    # - Interpretation: Easier to understand results
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    # X_2d shape: (n_samples, 2) - ready for 2D visualization

    # STEP 7: ISOLATION FOREST ANOMALY DETECTION
    # -------------------------------------------
    # LEARNING: Isolation Forest is an ensemble method
    # It builds multiple random trees and combines their results
    
    # Parameters:
    # - contamination: Expected proportion of anomalies (0.1 = 10%)
    # - random_state: Seed for reproducibility
    #
    # How it works:
    # 1. Build random binary trees (like decision trees)
    # 2. For each point, measure path length (splits needed to isolate it)
    # 3. Short path = anomaly (easy to isolate from others)
    # 4. Long path = normal (hard to isolate, similar to many points)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # LEARNING: fit_predict trains model and predicts anomalies
    # Returns: -1 for anomalies, 1 for normal points
    anomaly_pred = iso_forest.fit_predict(X_scaled)
    
    # LEARNING: score_samples gives continuous anomaly scores
    # Lower score = more anomalous (easier to isolate)
    # Higher score = more normal (harder to isolate)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    # Scores typically range from -0.5 to 0.5
    # Negative = anomaly, positive = normal

    # STEP 8: CONVERT TO BINARY FLAGS
    # --------------------------------
    # LEARNING: Isolation Forest returns -1 for anomalies, 1 for normal
    # Convert to boolean for easier interpretation
    is_anomaly = (anomaly_pred == -1).tolist()
    n_anomalies = sum(is_anomaly)

    # LEARNING: Results interpretation:
    # - points: 2D coordinates for visualization (colored by anomaly status)
    # - anomaly_scores: Continuous scores (lower = more anomalous)
    # - is_anomaly: Binary flags (True = anomaly, False = normal)
    # - n_anomalies: Count of detected anomalies
    
    return {
        'points': [[float(x), float(y)] for x, y in X_2d],
        'anomaly_scores': anomaly_scores.tolist(),
        'is_anomaly': is_anomaly,
        'n_anomalies': int(n_anomalies)
    }
