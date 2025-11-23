"""SHAP explainability for astronomy models."""
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def compute_shap_values(model_dict, data, feature_names):
    """Compute SHAP values."""
    if not SHAP_AVAILABLE or not model_dict or 'model' not in model_dict:
        return {
            'feature_names': feature_names,
            'values': [[0.0] * len(feature_names) for _ in range(len(data))],
            'base_value': 0.0
        }
    
    model = model_dict['model']
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        return {
            'feature_names': feature_names,
            'values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
            'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
        }
    except:
        return {
            'feature_names': feature_names,
            'values': [[0.0] * len(feature_names) for _ in range(len(data))],
            'base_value': 0.0
        }

