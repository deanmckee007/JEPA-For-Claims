# data/preprocessing.py
import pandas as pd
import numpy as np

def load_data(training_parquet_path):
    pd_training_df = pd.read_parquet(training_parquet_path)
    
    return pd_training_df

def filter_out_na_target(df):
    return df.dropna(subset=['target'])

def filter_rows_with_min_ttnc_tokens(df, min_ttnc_tokens=3):
    def count_ttnc_tokens(sequence):
        return sequence.count('ttnc_')
    
    filtered_df = df[df['input'].apply(count_ttnc_tokens) >= min_ttnc_tokens]
    
    return filtered_df

def transform_target(df):
    print('percent < 1500', np.mean(df.target<=1500))
    print('percent 1500-4500', np.mean( (df.target>1500) & (df.target<=4500)))
    print('percent 4500-7500', np.mean( (df.target>4500) & (df.target<=7500)))
    print('percent > 7500', np.mean(df.target>7500))
    df.loc[:, 'target'] = df['target'].fillna(0)
    df['orig_target'] = df['target']
    df.loc[:, 'target'] = np.log1p(df['target'])
    df.loc[:, 'target'] = np.minimum(df['target'], 10)  # Cap the logged target at 10
    return df

def tokenize_input(sequence):
    if isinstance(sequence, str):
        return sequence.strip().split()
    return []

def remove_trailing_time_token(sequence):
    if sequence and sequence[-1].startswith('ttnc_'):
        return sequence[:-1]
    return sequence
