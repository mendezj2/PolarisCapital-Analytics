"""Schema inference for ingestion."""
import json
import sys
import pandas as pd
from api import common_preprocess

def infer_schema_from_csv(path, sample_rows=5000):
    """Infer schema from CSV."""
    df = pd.read_csv(path, nrows=sample_rows)
    schema = common_preprocess.infer_schema(df)
    schema['domain_guess'] = common_preprocess.detect_domain_from_columns(df.columns)
    return schema

def save_schema(schema, destination):
    """Save schema to JSON."""
    with open(destination, 'w') as f:
        json.dump(schema, f, indent=2)

if __name__ == '__main__':
    input_csv = sys.argv[1] if len(sys.argv) > 1 else 'sample.csv'
    output_json = sys.argv[2] if len(sys.argv) > 2 else 'schema.json'
    
    schema = infer_schema_from_csv(input_csv)
    save_schema(schema, output_json)
    print(f"Schema saved to {output_json}")

