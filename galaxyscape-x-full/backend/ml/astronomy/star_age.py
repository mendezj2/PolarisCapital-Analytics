"""
Stellar Age Prediction using Gradient Boosting
===============================================

LEARNING: Supervised Learning - Regression
--------------------------------------------

This module demonstrates Gradient Boosting Regressor, a powerful ensemble
method for predicting continuous values (regression).

THEORY: What is Supervised Learning?
-----------------------------------
Supervised learning uses labeled examples to learn patterns:
- Input: Features (temperature, mass, radius, etc.)
- Output: Target (stellar age)
- Goal: Learn function f(features) → age

Why predict stellar age?
- Direct measurement is difficult (stars are far away)
- Can estimate from observable properties
- Useful for understanding stellar evolution

GRADIENT BOOSTING ALGORITHM:
---------------------------
Gradient Boosting is an ensemble method that combines weak learners:

1. Start with simple model (e.g., mean of target)
2. Fit new model to residuals (errors of previous model)
3. Add new model to ensemble (weighted combination)
4. Repeat steps 2-3 many times (n_estimators)

Key Insight: Each new model corrects mistakes of previous models!

MATHEMATICAL FOUNDATION:
-----------------------
Gradient Boosting minimizes loss function using gradient descent:

    F_m(x) = F_{m-1}(x) + α × h_m(x)

Where:
    - F_m = Ensemble model at iteration m
    - h_m = Weak learner (decision tree) at iteration m
    - α = Learning rate (step size)

Loss function (for regression): L(y, F(x)) = ½(y - F(x))²
Gradient: ∂L/∂F = -(y - F(x)) = -residual

LEARNING: Each tree fits the negative gradient (residuals)!

HYPERPARAMETERS:
---------------
- n_estimators: Number of trees (more = better fit, risk of overfitting)
- max_depth: Maximum tree depth (controls complexity)
- learning_rate: Step size (smaller = slower but more stable)

EVALUATION METRICS:
------------------
1. MAE (Mean Absolute Error):
   MAE = (1/n) × Σ|y_i - ŷ_i|
   
   Interpretation: Average prediction error in same units as target
   Example: MAE = 1.5 billion years means average error is 1.5 Gyr

2. RMSE (Root Mean Squared Error):
   RMSE = √[(1/n) × Σ(y_i - ŷ_i)²]
   
   Interpretation: Penalizes large errors more than MAE
   Always ≥ MAE (due to squaring)

3. R² (Coefficient of Determination):
   R² = 1 - (SS_res / SS_tot)
   
   Interpretation: Proportion of variance explained
   R² = 0.8 means model explains 80% of variance

FEATURE ENGINEERING:
-------------------
Select numeric features that correlate with age:
- Temperature: Hotter stars often younger
- Mass: More massive stars burn faster (shorter lifetime)
- Luminosity: Related to mass and age
- Rotation period: Slows with age

LEARNING CHECKPOINT:
-------------------
1. Why use ensemble methods instead of single model?
   → Combining multiple models reduces error (wisdom of crowds)

2. What does MAE = 2.0 billion years mean?
   → Average prediction error is 2 billion years

3. Why fit trees to residuals?
   → Each tree corrects mistakes of previous trees
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ml.data_loaders import load_astronomy_data


def get_star_age_predictions():
    """
    Predict stellar age using Gradient Boosting Regressor.
    
    LEARNING: This function demonstrates:
    1. Supervised learning pipeline (features → target)
    2. Feature selection (choosing relevant features)
    3. Model training (Gradient Boosting)
    4. Model evaluation (MAE, RMSE, R²)
    5. Prediction generation (actual vs predicted)
    
    Returns:
        dict with actual, predicted, features, mae, rmse, r2
    """
    # STEP 1: DATA LOADING
    # ---------------------
    df = load_astronomy_data()
    if df is None or len(df) == 0:
        return {
            'actual': [],
            'predicted': [],
            'features': [],
            'mae': 0.0,
            'rmse': 0.0,
            'r2': 0.0
        }

    # STEP 2: TARGET VARIABLE SELECTION
    # ----------------------------------
    # LEARNING: Target = what we want to predict (stellar age)
    # Try different column names (datasets use different names)
    target_col = next((c for c in ['age', 'stellar_age', 'star_age'] if c in df.columns), None)
    
    if target_col is None:
        # LEARNING: If no age data, create synthetic target for demonstration
        # In real scenario, would need actual age measurements
        df['age'] = np.random.uniform(1, 10, len(df)) * 1e9  # Ages in years (billions)
        target_col = 'age'

    # STEP 3: FEATURE SELECTION
    # --------------------------
    # LEARNING: Features = inputs to model (what we observe)
    # Select numeric columns (temperature, mass, radius, etc.)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # LEARNING: Remove target and other non-feature columns
    # Don't want to predict age using age itself!
    for col in [target_col, 'cluster', 'id']:
        if col in feature_cols:
            feature_cols.remove(col)
    
    # Limit to top 10 features (avoid curse of dimensionality)
    feature_cols = feature_cols[:10]

    # STEP 4: HANDLE INSUFFICIENT FEATURES
    # ------------------------------------
    if len(feature_cols) < 2:
        # Generate synthetic features if needed
        df['feat1'] = np.random.randn(len(df))
        df['feat2'] = np.random.randn(len(df))
        feature_cols = ['feat1', 'feat2']

    # STEP 5: PREPARE FEATURE MATRIX AND TARGET
    # -----------------------------------------
    # LEARNING: X = feature matrix (n_samples × n_features)
    #          y = target vector (n_samples)
    X = df[feature_cols].fillna(df[feature_cols].median())  # Fill missing with median
    y = df[target_col].fillna(df[target_col].median())  # Fill missing target with median

    # STEP 6: TRAIN GRADIENT BOOSTING MODEL
    # --------------------------------------
    # LEARNING: Gradient Boosting is an ensemble method
    # Combines multiple decision trees to make predictions
    
    # Hyperparameters:
    # - n_estimators: Number of trees (100 = build 100 trees)
    # - max_depth: Maximum tree depth (5 = trees up to 5 levels deep)
    # - random_state: Seed for reproducibility
    #
    # LEARNING: How Gradient Boosting works:
    # 1. Start with simple prediction (mean of target)
    # 2. Fit tree to residuals (errors)
    # 3. Add tree to ensemble (weighted combination)
    # 4. Repeat steps 2-3 n_estimators times
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    # LEARNING: fit() trains the model
    # Model learns: f(temperature, mass, radius, ...) → age
    model.fit(X, y)
    
    # STEP 7: GENERATE PREDICTIONS
    # -----------------------------
    # LEARNING: predict() applies learned function to features
    # Returns predicted age for each star
    y_pred = model.predict(X)
    # Shape: (n_samples,) - one prediction per star

    # STEP 8: EVALUATE MODEL PERFORMANCE
    # -----------------------------------
    # LEARNING: Compare predictions to actual values
    
    # MAE (Mean Absolute Error):
    # LEARNING: Average absolute difference between actual and predicted
    # Formula: MAE = (1/n) × Σ|y_i - ŷ_i|
    # Units: Same as target (e.g., billion years)
    # Interpretation: On average, predictions are off by MAE units
    mae = mean_absolute_error(y, y_pred)
    
    # RMSE (Root Mean Squared Error):
    # LEARNING: Penalizes large errors more than MAE
    # Formula: RMSE = √[(1/n) × Σ(y_i - ŷ_i)²]
    # Always ≥ MAE (due to squaring)
    # Interpretation: Typical prediction error (larger errors weighted more)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # R² (Coefficient of Determination):
    # LEARNING: Proportion of variance explained by model
    # Formula: R² = 1 - (SS_res / SS_tot)
    # Range: -∞ to 1.0 (1.0 = perfect, 0 = no better than mean)
    # Interpretation: R² = 0.8 means model explains 80% of variance
    r2 = r2_score(y, y_pred)

    # STEP 9: SAMPLE POINTS FOR VISUALIZATION
    # ----------------------------------------
    # LEARNING: Don't return all points (too many for frontend)
    # Randomly sample subset for scatter plot visualization
    n_points = min(100, len(y))
    indices = np.random.choice(len(y), n_points, replace=False)

    # LEARNING: Results interpretation:
    # - actual: True stellar ages (ground truth)
    # - predicted: Model's age predictions
    # - features: Which features model used
    # - mae: Average prediction error (lower is better)
    # - rmse: Typical prediction error, penalizes large errors (lower is better)
    # - r2: Model quality (higher is better, max = 1.0)
    
    return {
        'actual': y.iloc[indices].tolist() if isinstance(y, pd.Series) else y[indices].tolist(),
        'predicted': y_pred[indices].tolist(),
        'features': feature_cols,
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }
