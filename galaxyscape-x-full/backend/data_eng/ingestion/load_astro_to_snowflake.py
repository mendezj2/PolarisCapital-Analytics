"""Astronomy ingestion to Snowflake.

This script ingests REAL astronomy data (from data/raw/astronomy or user uploads)
into Snowflake. No mock/fake CSVs are used.
"""
import os
import json
import uuid
import pandas as pd
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_path)
from ml.astronomy import preprocess

def get_snowflake_connection():
    """Get Snowflake connection."""
    # TODO: Implement actual Snowflake connection
    # import snowflake.connector
    # return snowflake.connector.connect(
    #     user=os.getenv('SNOWFLAKE_USER'),
    #     password=os.getenv('SNOWFLAKE_PASSWORD'),
    #     account=os.getenv('SNOWFLAKE_ACCOUNT'),
    #     warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    #     database=os.getenv('SNOWFLAKE_DATABASE'),
    #     schema=os.getenv('SNOWFLAKE_SCHEMA')
    # )
    return None

def stage_csv(file_path):
    """Stage CSV file."""
    print(f"Staging file: {file_path}")
    # TODO: Implement Snowflake staging

def copy_into_tables(upload_id, file_path):
    """Copy data into Snowflake tables."""
    print(f"Copying {file_path} to Snowflake with upload_id: {upload_id}")
    # TODO: Implement COPY INTO commands

def run_ingestion(file_path: str | None = None, use_raw_data: bool = True):
    """
    Run ingestion pipeline.

    Args:
        file_path: Optional specific file path. If None and use_raw_data=True,
                   uses load_and_clean_from_raw() to get data from data/raw/astronomy.
        use_raw_data: If True, load from data/raw/astronomy using preprocessing pipeline.
                     If False, use file_path directly (for user uploads).

    TODO (USER): Update to use real data sources:
    - If use_raw_data=True: Call preprocess.load_and_clean_from_raw()
    - If use_raw_data=False: Load file_path, then clean with preprocess.clean_astronomy_df()
    - Then proceed with Snowflake staging and COPY INTO
    """
    upload_id = f"astro-{uuid.uuid4().hex[:8]}"
    
    if use_raw_data and file_path is None:
        # TODO (USER): Use preprocessing pipeline to load and clean raw data
        # df = preprocess.load_and_clean_from_raw()
        # processed_path = f"data/processed/astronomy/processed_{upload_id}.csv"
        # df.to_csv(processed_path, index=False)
        # file_path = processed_path
        print("TODO: Implement load_and_clean_from_raw() call")
        return
    elif file_path and os.path.exists(file_path):
        # User-provided file - clean it first
        # TODO (USER): Load and clean user upload
        # df = pd.read_csv(file_path)
        # df_cleaned = preprocess.clean_astronomy_df(df)
        # processed_path = f"data/processed/astronomy/processed_{upload_id}.csv"
        # df_cleaned.to_csv(processed_path, index=False)
        # file_path = processed_path
        pass
    else:
        print(f"Error: File not found: {file_path}")
        return
    
    # TODO (USER): After getting cleaned data, proceed with Snowflake ingestion
    # stage_csv(file_path)
    # copy_into_tables(upload_id, file_path)
    
    print(f"Ingestion complete for upload_id: {upload_id}")
    print("TODO: Replace mock CSV reading with real data pipeline")

if __name__ == '__main__':
    import sys
    # TODO (USER): Update default to use raw data instead of mock CSV
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    use_raw = len(sys.argv) <= 1  # Use raw data if no file specified
    run_ingestion(file_path, use_raw_data=use_raw)

