"""
Thin compatibility layer that re-exports finance ML helpers.

Each algorithm now lives in its own module for clearer organization.
"""
from ml.finance.risk_metrics import get_risk_metrics
from ml.finance.correlation_network import get_correlation_network
from ml.finance.future_outcomes import get_future_outcomes
from ml.finance.portfolio_stats import get_portfolio_stats
from ml.finance.streaming_metrics import get_streaming_metrics

__all__ = [
    'get_risk_metrics',
    'get_correlation_network',
    'get_future_outcomes',
    'get_portfolio_stats',
    'get_streaming_metrics',
]
