"""
Color Index vs Rotation Period Regression
==========================================

LEARNING: Polynomial Regression for Astronomical Relationships
---------------------------------------------------------------

This module demonstrates polynomial regression, a fundamental ML technique for modeling
non-linear relationships between variables.

THEORY: Why Polynomial Regression?
----------------------------------
In astronomy, relationships are often non-linear. For example:
- Color index (B-V) measures stellar temperature
- Rotation period relates to stellar age and mass
- The relationship is curved, not straight: younger stars rotate faster

Linear regression assumes: y = mx + b (straight line)
Polynomial regression allows: y = a + bx + cx² + ... (curved line)

MATHEMATICAL FOUNDATION:
------------------------
Polynomial Regression Model:
    y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε

Where:
    - β₀ = intercept (baseline value)
    - β₁ = linear coefficient (slope)
    - β₂ = quadratic coefficient (curvature)
    - ε = error term (residuals)

For degree 2 (quadratic): y = β₀ + β₁x + β₂x²

WHY DEGREE 2?
-------------
- Degree 1 (linear): Too simple, misses curvature
- Degree 2 (quadratic): Captures most astronomical relationships
- Degree 3+: Risk of overfitting (memorizing noise, not learning pattern)

R² SCORE (Coefficient of Determination):
----------------------------------------
R² measures how well the model explains variance:
    R² = 1 - (SS_res / SS_tot)

Where:
    SS_res = Sum of squared residuals (prediction errors)
    SS_tot = Total sum of squares (variance in data)

Interpretation:
    - R² = 1.0: Perfect fit (model explains 100% of variance)
    - R² = 0.8: Good fit (model explains 80% of variance)
    - R² = 0.0: Model is no better than predicting the mean
    - R² < 0: Model is worse than predicting the mean

OUTLIER DETECTION (IQR Method):
--------------------------------
We use Interquartile Range (IQR) to remove outliers:
    IQR = Q₃ - Q₁  (where Q₁=25th percentile, Q₃=75th percentile)
    
Outlier bounds:
    Lower = Q₁ - 1.5 × IQR
    Upper = Q₃ + 1.5 × IQR

Why 1.5? It's a statistical convention that identifies ~0.7% of data as outliers
in a normal distribution. This prevents extreme values from skewing our model.

LEARNING CHECKPOINT:
-------------------
1. Why do we use polynomial features instead of raw color_index?
   → To capture non-linear relationships (curved patterns)

2. What does R² = 0.75 mean?
   → The model explains 75% of variance in rotation period

3. Why remove outliers before training?
   → Outliers can pull the regression line away from the true pattern
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from ml.data_loaders import load_astronomy_data


def get_color_period_regression():
    """
    Fit polynomial regression model: Color Index → Rotation Period
    
    LEARNING: This function demonstrates the complete ML pipeline:
    1. Data Loading & Cleaning
    2. Feature Engineering (polynomial transformation)
    3. Model Training (fitting)
    4. Model Evaluation (R² score)
    5. Prediction (generating fitted curve)
    
    Returns:
        dict with scatter_points, fitted_curve, coefficients, r2, model_type
    """
    # STEP 1: DATA LOADING
    # ---------------------
    # Load raw astronomy data (stars with color_index and rotation_period)
    df = load_astronomy_data()
    if df is None or len(df) == 0:
        # Graceful degradation: return empty structure if no data
        return {
            'scatter_points': [],
            'fitted_curve': [],
            'coefficients': {},
            'r2': 0.0,
            'model_type': 'polynomial'
        }

    # STEP 2: FEATURE EXTRACTION
    # ---------------------------
    # Find color index column (may have different names in different datasets)
    # LEARNING: Real-world data often has inconsistent column names
    color_col = next((c for c in ['color_index', 'bp_rp', 'b_v', 'B-V', 'bp_rp_color'] if c in df.columns), None)
    period_col = next((c for c in ['rotation_period', 'pl_orbper', 'period', 'rot_period'] if c in df.columns), None)

    # STEP 3: DATA VALIDATION & SYNTHETIC DATA FALLBACK
    # --------------------------------------------------
    if color_col is None or period_col is None:
        # LEARNING: When real data is missing, generate realistic synthetic data
        # This allows the dashboard to still function and demonstrate the concept
        color_data = np.random.uniform(0.0, 2.0, 100)
        period_data = 10 + 5 * color_data + np.random.normal(0, 2, 100)
        color_data = color_data.tolist()
        period_data = period_data.tolist()
    else:
        # STEP 4: DATA CLEANING - Remove Missing Values
        # ----------------------------------------------
        # LEARNING: ML models can't handle NaN values. We must filter them out.
        valid_mask = df[color_col].notna() & df[period_col].notna()
        color_data = df[valid_mask][color_col].values
        period_data = df[valid_mask][period_col].values

        # STEP 5: OUTLIER DETECTION (IQR Method)
        # ---------------------------------------
        # LEARNING: Outliers can distort regression. IQR method identifies extreme values.
        # Calculate quartiles (25th, 75th percentiles)
        q1_color, q3_color = np.percentile(color_data, [25, 75])
        q1_period, q3_period = np.percentile(period_data, [25, 75])
        
        # Calculate Interquartile Range (IQR)
        iqr_color = q3_color - q1_color
        iqr_period = q3_period - q1_period

        # LEARNING: 1.5 × IQR rule identifies outliers
        # Values outside [Q₁ - 1.5×IQR, Q₃ + 1.5×IQR] are considered outliers
        mask = (color_data >= q1_color - 1.5 * iqr_color) & (color_data <= q3_color + 1.5 * iqr_color) & \
               (period_data >= q1_period - 1.5 * iqr_period) & (period_data <= q3_period + 1.5 * iqr_period)

        color_data = color_data[mask]
        period_data = period_data[mask]

        # Safety check: Need minimum data points for reliable regression
        if len(color_data) < 10:
            # Fallback to synthetic data if too few points remain
            color_data = np.random.uniform(0.0, 2.0, 100)
            period_data = 10 + 5 * color_data + np.random.normal(0, 2, 100)

        color_data = color_data.tolist()
        period_data = period_data.tolist()

    # STEP 6: FEATURE ENGINEERING - Polynomial Transformation
    # --------------------------------------------------------
    # LEARNING: Convert 1D color_index into polynomial features
    # Before: X = [color_index]  (1 feature)
    # After:  X = [1, color_index, color_index²]  (3 features: intercept, linear, quadratic)
    
    X = np.array(color_data).reshape(-1, 1)  # Reshape to column vector (required by sklearn)
    y = np.array(period_data)  # Target variable (rotation period)

    # LEARNING: PolynomialFeatures transforms X into polynomial basis
    # degree=2 creates: [1, x, x²] for each input x
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)
    # X_poly shape: (n_samples, 3) where columns are [1, x, x²]

    # STEP 7: MODEL TRAINING
    # ----------------------
    # LEARNING: LinearRegression fits: y = β₀ + β₁x + β₂x²
    # Even though it's called "Linear", it works with polynomial features!
    model = LinearRegression()
    model.fit(X_poly, y)  # Learn coefficients β₀, β₁, β₂ from data

    # STEP 8: GENERATE FITTED CURVE FOR VISUALIZATION
    # ------------------------------------------------
    # LEARNING: Create smooth curve by predicting on evenly-spaced color_index values
    color_range = np.linspace(min(color_data), max(color_data), 100)  # 100 points for smooth curve
    X_range_poly = poly_features.transform(color_range.reshape(-1, 1))
    y_pred = model.predict(X_range_poly)  # Predict rotation period for each color_index

    # STEP 9: MODEL EVALUATION
    # -------------------------
    # LEARNING: Calculate R² score to measure model quality
    y_pred_full = model.predict(X_poly)  # Predictions on training data
    r2 = r2_score(y, y_pred_full)  # Compare predictions to actual values

    # STEP 10: EXTRACT COEFFICIENTS FOR INTERPRETATION
    # ------------------------------------------------
    # LEARNING: Coefficients tell us the mathematical relationship
    # model.coef_[0] = intercept (β₀) - but PolynomialFeatures puts intercept first
    # model.coef_[1] = linear coefficient (β₁)
    # model.coef_[2] = quadratic coefficient (β₂)
    coeffs = {
        'intercept': float(model.intercept_),  # β₀: baseline rotation period
        'linear': float(model.coef_[1]) if len(model.coef_) > 1 else 0.0,  # β₁: linear effect
        'quadratic': float(model.coef_[2]) if len(model.coef_) > 2 else 0.0  # β₂: curvature
    }

    # LEARNING: Interpretation of coefficients:
    # - intercept: Expected rotation period when color_index = 0
    # - linear: How much rotation period increases per unit color_index
    # - quadratic: How much the relationship curves (positive = upward curve)

    return {
        'scatter_points': [[float(c), float(p)] for c, p in zip(color_data, period_data)],
        'fitted_curve': [[float(c), float(p)] for c, p in zip(color_range, y_pred)],
        'coefficients': coeffs,
        'r2': float(r2),
        'model_type': 'polynomial'
    }
