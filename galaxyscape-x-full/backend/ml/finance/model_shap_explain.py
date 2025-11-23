"""SHAP explanations for finance."""
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def compute_risk_shap(model_dict, data):
    """Compute SHAP values for risk model."""
    if not SHAP_AVAILABLE or not model_dict or 'model' not in model_dict:
        return {'values': [], 'base_value': 0.0}
    
    model = model_dict['model']
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        
        return {
            'values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
            'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
        }
    except:
        return {'values': [], 'base_value': 0.0}

