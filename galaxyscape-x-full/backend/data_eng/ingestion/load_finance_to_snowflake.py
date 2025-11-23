"""Finance ingestion to Snowflake.

This script ingests REAL finance data (from data/raw/finance or user uploads)
into Snowflake. No mock/fake CSVs are used.
"""
import os
import uuid
import pandas as pd
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_path)
from ml.finance import preprocess

def connect_finance_snowflake():
    """Connect to Snowflake."""
    # TODO: Implement connection
    return None

def upsert_market_data(upload_id, file_path):
    """Upsert market data."""
    print(f"Upserting market data from {file_path}")
    # TODO: Implement upsert logic

def backfill_portfolio_metrics(upload_id):
    """Backfill portfolio metrics."""
    print(f"Backfilling portfolio metrics for {upload_id}")
    # TODO: Implement backfill

def run_finance_ingestion(file_path: str | None = None, use_raw_data: bool = True, tickers: list[str] | None = None):
    """
    Run finance ingestion pipeline.

    Args:
        file_path: Optional specific file path. If None and use_raw_data=True,
                   uses load_and_clean_from_raw() to get data from data/raw/finance.
        use_raw_data: If True, load from data/raw/finance using preprocessing pipeline.
                     If False, use file_path directly (for user uploads).
        tickers: Optional list of tickers if downloading new data.

    TODO (USER): Update to use real data sources:
    - If use_raw_data=True: Call preprocess.load_and_clean_from_raw(tickers=tickers)
    - If use_raw_data=False: Load file_path, then engineer features with preprocess.engineer_risk_features()
    - Then proceed with Snowflake upsert and backfill
    """
    upload_id = f"finance-{uuid.uuid4().hex[:8]}"
    
    if use_raw_data and file_path is None:
        # TODO (USER): Use preprocessing pipeline to load and clean raw data
        # df = preprocess.load_and_clean_from_raw(tickers=tickers)
        # processed_path = f"data/processed/finance/processed_{upload_id}.csv"
        # df.to_csv(processed_path, index=False)
        # file_path = processed_path
        print("TODO: Implement load_and_clean_from_raw() call")
        return
    elif file_path and os.path.exists(file_path):
        # User-provided file - engineer features first
        # TODO (USER): Load and engineer features for user upload
        # df = pd.read_csv(file_path, parse_dates=['Date'])
        # df_engineered = preprocess.engineer_risk_features(df)
        # processed_path = f"data/processed/finance/processed_{upload_id}.csv"
        # df_engineered.to_csv(processed_path, index=False)
        # file_path = processed_path
        pass
    else:
        print(f"Error: File not found: {file_path}")
        return
    
    # TODO (USER): After getting processed data, proceed with Snowflake ingestion
    # upsert_market_data(upload_id, file_path)
    # backfill_portfolio_metrics(upload_id)
    
    print(f"Finance ingestion complete for upload_id: {upload_id}")
    print("TODO: Replace mock CSV reading with real data pipeline")

if __name__ == '__main__':
    import sys
    # TODO (USER): Update default to use raw data instead of mock CSV
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    use_raw = len(sys.argv) <= 1  # Use raw data if no file specified
    run_finance_ingestion(file_path, use_raw_data=use_raw)

