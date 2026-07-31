import os
import scanpy as sc
import numpy as np
import pandas as pd
import harmonypy as hm
import logging
from preprocessing import read_flow, pd_to_adata

FLOW_COFACTOR = 150
MAX_SCALE_VALUE = 3
STRATIFIED_CELL_MAX = 50000


EXCLUDE_CHANNELS = [
    'FSC-A', 'FSC-H', 'SSC-A', 'SSC-B-A', 'SSC-B-H', 'SSC-H',
    'AF-A', 'CD66bCD19CD326LD', 'Time', 'CD45', 'Event #'
]

DATA_DIRECTORIES = {
    'BOM_CD3_01DEC25': 'BOM',
    'LLN_CD3_01DEC25': 'LLN',
    'LNG_CD3_01DEC25': 'LNG',
    'MLN_CD3_01DEC25': 'MLN',
    'SPL_CD3_01DEC25': 'SPL'
}


def load_and_merge_cohorts(data_map, exclude_cols):
    """
    Loads flow cytometry data from multiple directories, merges them into a single AnnData object,
    and excludes specified columns.

    Parameters
    ----------
    data_map : dict
        A dictionary mapping directory paths to tissue labels.
    exclude_cols : list
        A list of column names to exclude from the final AnnData object.

    Returns
    -------
    anndata.AnnData
        An AnnData object containing the merged flow cytometry data, with specified columns excluded.
    """

    flow_dfs = []
    for directory, tissue_label in data_map.items():
        if os.path.exists(directory):
            df, _, _ = read_flow(directory, tissue=tissue_label)
            flow_dfs.append(df)
        else:
            logging.warning(
                f"Data pathway skip: Directory '{directory}' not found locally.")

    if not flow_dfs:
        raise FileNotFoundError(
            "Zero clinical tissue cohorts successfully parsed into memory.")

    df_all_tissues = pd.concat(flow_dfs, ignore_index=True)
    df_counts = df_all_tissues.drop(columns=exclude_cols, errors='ignore')
    df_counts = df_counts.select_dtypes(include=[np.number]).copy()

    if 'age' in df_counts.columns:
        df_counts.drop(columns=['age'], inplace=True)

    return pd_to_adata(df_all_tissues, df_counts)
