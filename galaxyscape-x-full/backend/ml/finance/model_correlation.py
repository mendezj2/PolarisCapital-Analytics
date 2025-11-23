"""Correlation analytics."""
import pandas as pd
import numpy as np

def compute_correlation_matrix(df, value_col='return'):
    """Compute correlation matrix."""
    if value_col not in df.columns:
        # Use numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return df[numeric_cols].corr()
        return pd.DataFrame()
    
    # Pivot if needed
    if 'ticker' in df.columns and 'date' in df.columns:
        pivoted = df.pivot_table(index='date', columns='ticker', values=value_col)
        return pivoted.corr()
        return pivoted.corr()
    
    return pd.DataFrame()

def detect_sector_clusters(corr_matrix, metadata):
    """Detect sector clusters from correlation."""
    if corr_matrix.empty:
        return []
    
    # Simple clustering based on correlation threshold
    clusters = []
    processed = set()
    
    for ticker in corr_matrix.index:
        if ticker in processed:
            continue
        
        cluster = [ticker]
        for other_ticker in corr_matrix.columns:
            if other_ticker != ticker and other_ticker not in processed:
                corr = corr_matrix.loc[ticker, other_ticker]
                if corr > 0.7:  # High correlation threshold
                    cluster.append(other_ticker)
                    processed.add(other_ticker)
        
        processed.add(ticker)
        if len(cluster) > 1:
            clusters.append({
                'tickers': cluster,
                'size': len(cluster),
                'avg_correlation': np.mean([corr_matrix.loc[c1, c2] 
                                           for i, c1 in enumerate(cluster) 
                                           for c2 in cluster[i+1:]])
            })
    
    return clusters

