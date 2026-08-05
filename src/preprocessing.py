import logging

import flowkit as fk
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def read_flow(directory, tissue=None):
    """
    Reads flow cytometry data from a specified directory and returns a DataFrame containing the data,
    a list of sample IDs, and the FlowKit session object.

    Parameters:
    -----------
    directory : str
        The path to the directory containing flow cytometry data files.
    tissue : str, optional
        The type of tissue being analyzed, used to annotate the DataFrame.       

    Returns:
    --------
    tuple
        A tuple containing:
        - df_flow : pandas.DataFrame
            A DataFrame containing flow cytometry data with additional metadata columns.
        - sample_list : list
            A list of sample IDs extracted from the FlowKit session.
        - session : flowkit.Session
            The FlowKit session object used to read the flow cytometry data.
    """

    try:
        session = fk.Session(fcs_samples=directory)
    except Exception as e:
        raise IOError(
            f"Failed to initialize FlowKit Session on directory '{directory}': {e}")

    sample_list = session.get_sample_ids()
    df_flow = []

    for sample_id in sample_list:
        df = session.get_gate_events(sample_id)
        if tissue is not None:
            df['tissue'] = tissue
        try:
            df['age'] = int(sample_id.split('_')[2])
        except (IndexError, ValueError):
            logging.warning(
                f"Could not extract age from sample ID '{sample_id}'. Setting age to NaN.")
            df['age'] = np.nan
        df["sample_id"] = sample_id

        if "ctr" in sample_id.lower():
            df["condition"] = "ctr"
            df["asthma"] = "control"
        elif "hst" in sample_id.lower():
            df["condition"] = "hst"
            df["asthma"] = "asthmatic"
        elif "ftl" in sample_id.lower():
            df["condition"] = "ftl"
            df["asthma"] = "asthmatic"
        else:
            raise ValueError(
                f"Critical Metadata Discrepancy: Sample ID '{sample_id}' does not match "
                f"standard Farber Lab cohorts ('ctr', 'hst', 'ftl'). Fix raw naming conventions."
            )

        df_flow.append(df)

    if not df_flow:
        raise ValueError(
            f"No valid data frames generated from directory: {directory}")

    df_flow = pd.concat(df_flow)
    df_flow.columns = [pns if pns !=
                       '' else pnn for pnn, pns in df_flow.columns]
    logging.info(
        f"Successfully processed {len(sample_list)} files for tissue: {tissue}")
    return df_flow, sample_list, session


def automate_merging(df, columns, threshold=0.6):
    """
    Merges duplicate channels in dataframe as a result of utilizing different fluorophores for the same marker.
    Updates dataframe with merged channels for each duplicate marker.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing flow cytometry data, containing duplicate channels.
    columns : list
        List containing the columns that are duplicate channels.

    Returns:
    --------
    pandas.DataFrame
        Dataframe containing flow cytometry data with merged channels and excluding duplicates.

    """
    remaining = set(columns)
    while remaining:
        s1 = remaining.pop()
        matches = [(s2, SequenceMatcher(None, s1, s2).ratio())
                   for s2 in remaining]

        if matches:
            best_match, score = max(matches, key=lambda x: x[1])
            if score >= threshold:
                df[s1] = df[[s1, best_match]].max(axis=1)
                logging.info(
                    f"Automated Channel Merge: {best_match} -> {s1} (String Similarity Score: {score:.2f})")
                remaining.remove(best_match)
                df.drop(columns=[best_match], inplace=True)
    return df


def pd_to_adata(df_flow, df_flow_counts):
    """
    Converts flow cytometry data from pandas DataFrames to an AnnData object.

    This function processes flow cytometry data and associated counts, creating an AnnData object
    with appropriate metadata. It truncates sample IDs, creates a metadata DataFrame, and assigns
    group and sample ID information to the AnnData object's observation annotations.

    Parameters:
    -----------
    df_flow : pandas.DataFrame
        A DataFrame containing flow cytometry data, including 'sample_id' and 'condition' columns.
    df_flow_counts : pandas.DataFrame
        A DataFrame containing intensity data for the flow cytometry samples.

    Returns:
    --------
    anndata.AnnData
        An AnnData object containing the flow cytometry intensity data with associated metadata.
        The object includes:
        - X: The count matrix from df_flow_counts
        - obs: Observation annotations including 'group' and 'sample_id'

    """
    if df_flow_counts.isna().any().any():
        nan_cols = df_flow_counts.columns[df_flow_counts.isna().any()].tolist()
        df_flow_counts = automate_merging(df_flow_counts, nan_cols)

    short_sample_ids = df_flow['sample_id'].astype(str).str[:4]

    df_metadata = pd.DataFrame({
        'group': df_flow['condition'].values,
        'sample_id': short_sample_ids.values,
        'age': df_flow['age'].values,
        'asthma': df_flow['asthma'].values
    }, index=df_flow_counts.index)

    if 'tissue' in df_flow.columns:
        df_metadata['tissue'] = df_flow.tissue

    adata = sc.AnnData(df_flow_counts)
    df_metadata.index = adata.obs.index

    for col in df_metadata.columns:
        adata.obs[col] = df_metadata[col]

    return adata


def pca_df(sample, session=None, singular=True, sample_list=None):
    """
    Creates a DataFrame of PCA values from flow cytometry data, grouped by sample ID.

    This function extracts PCA coordinates from an AnnData object, groups them by sample ID,
    and adds donor group information based on sample IDs. It can work with either a single
    sample or multiple samples.

    Parameters:
    -----------
    sample : anndata.AnnData
        AnnData object containing PCA coordinates in the obsm['X_pca'] attribute and
        sample IDs in obs['sample_id'].
    session : flowkit.Session, optional
        FlowKit session object used to retrieve sample IDs when singular=True.
    singular : bool, default=True
        If True, retrieves sample IDs from the session. If False, uses the provided sample_list.
    sample_list : list, optional
        List of sample IDs to use when singular=False.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing averaged PCA coordinates for each sample ID, with columns:
        - PC1, PC2, ...: Principal component coordinates
        - donor_group: Donor group information extracted from sample IDs
    """

    if singular:
        sample_list = session.get_sample_ids()

    sample_dict = {}
    for sample_id in sample_list:
        sample_dict[sample_id[:4]
                    ] = 'ctr' if 'ctr' in sample_id else 'hst' if 'hst' in sample_id else 'ftl'

    pca_df = pd.DataFrame(sample.obsm['X_pca'], columns=[
                          f'PC{i+1}' for i in range(sample.obsm['X_pca'].shape[1])])
    pca_df['sample_id'] = sample.obs['sample_id'].values

    grouped_pca = pca_df.groupby('sample_id').mean()
    grouped_pca['donor_group'] = grouped_pca.index.map(sample_dict)

    return grouped_pca


def population_filter(adata, population_column, population_value):
    """
    Filters the AnnData object based on a specified population column and value.

    Parameters:
    -----------
    adata : anndata.AnnData
        AnnData object containing flow cytometry data with observation annotations.
    population_column : str
        The name of the column in adata.obs to filter on (e.g., 'CD4', 'CD8').
    population_value : float
        The threshold value for filtering the specified population column.

    Returns:
    --------
    anndata.AnnData
        A filtered AnnData object containing only the cells that meet the specified population criteria.
    """

    if population_column not in adata.obs.columns:
        raise ValueError(
            f"Column '{population_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    if population_column == 'CD4':
        try:
            CD4 = adata[(adata[:, 'CD4'].X > population_value)]
        except KeyError:
            raise KeyError(
                f"Column 'CD4' not found in adata.var. Available columns: {list(adata.var.index)}")

        CD4 = CD4[:, CD4.var.index != 'gdTCR']
        CD4 = CD4[:, CD4.var.index != 'TCRva']
        CD4 = CD4[:, CD4.var.index != 'CD4']
        CD4 = CD4[:, CD4.var.index != 'CD8']

        groups = ['ctr', 'hst', 'ftl']
        tissues = CD4.obs['tissue'].unique().tolist()
        adata_list = []

        for group in groups:
            for tissue in tissues:
                subset = CD4[(CD4.obs['group'] == group) &
                             (CD4.obs['tissue'] == tissue)]
                if subset.shape[0] >= 5000:
                    sampled_subset = subset[np.random.choice(
                        subset.shape[0], 50000, replace=False)]
                else:
                    sampled_subset = subset

                sampled_subset.obs['group'] = f"{group}"
                adata_list.append(sampled_subset.copy())

        CD4 = sc.AnnData.concatenate(*adata_list, index_unique=None)
        return CD4

    elif population_column == 'CD8':
        try:
            CD8 = adata[(adata[:, 'CD8'].X > population_value)]
        except KeyError:
            raise KeyError(
                f"Column 'CD8' not found in adata.var. Available columns: {list(adata.var.index)}")

        CD8 = CD8[:, CD8.var.index != 'gdTCR']
        CD8 = CD8[:, CD8.var.index != 'TCRva']
        CD8 = CD8[:, CD8.var.index != 'CD4']
        CD8 = CD8[:, CD8.var.index != 'CD8']

        groups = ['ctr', 'hst', 'ftl']
        tissues = CD8.obs['tissue'].unique().tolist()
        adata_list = []

        for group in groups:
            for tissue in tissues:
                subset = CD8[(CD8.obs['group'] == group) &
                             (CD8.obs['tissue'] == tissue)]
                if subset.shape[0] >= 5000:
                    sampled_subset = subset[np.random.choice(
                        subset.shape[0], 50000, replace=False)]
                else:
                    sampled_subset = subset

                sampled_subset.obs['group'] = f"{group}"
                adata_list.append(sampled_subset.copy())

        CD8 = sc.AnnData.concatenate(*adata_list, index_unique=None)
        return CD8

    else:
        raise NotImplementedError(
            f"Population filtering for '{population_column}' is not implemented. Supported populations: 'CD4', 'CD8'.")
