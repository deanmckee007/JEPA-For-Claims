import pandas as pd


def _read_file(path: str) -> pd.DataFrame:
    """Read a dataframe from a parquet or pickle file based on extension."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if path.endswith('.pckl') or path.endswith('.pickle'):
        return pd.read_pickle(path)
    # fallback to csv
    return pd.read_csv(path)


def load_data(training_path: str, testing_path: str):
    """Load training and testing data from disk.

    Parameters
    ----------
    training_path : str
        Path to the training dataset.
    testing_path : str
        Path to the testing dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Loaded training and testing dataframes.
    """
    train_df = _read_file(training_path)
    test_df = _read_file(testing_path)
    return train_df, test_df
