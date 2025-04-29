import flowkit as fk 
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
def read_flow(directory):
    """
    Reads flow cytometry data from FCS files in a specified directory and processes it into a pandas DataFrame.

    This function creates a flowkit Session from the FCS files in the given directory, extracts events data
    for each sample, adds sample identification and condition information, and combines all samples into a
    single DataFrame.

    Parameters:
    -----------
    directory : str
        The path to the directory containing FCS files to be processed.

    Returns:
    --------
    tuple
        A tuple containing three elements:
        1. pandas.DataFrame: A DataFrame containing combined flow cytometry data from all samples in the directory.
           The DataFrame includes columns for sample ID, condition, and all parameters from the FCS files.
           Column names are adjusted to use parameter short names (pns) where available.
        2. list: A list of sample IDs processed from the FCS files.
        3. flowkit.Session: The flowkit Session object created from the FCS files.

    """

    session = fk.Session(fcs_samples=directory)
    sample_list = session.get_sample_ids()
    
    df_flow = []
    for sample_id in sample_list:
        df = session.get_gate_events(sample_id)
        df["sample_id"] = sample_id
        if "ctr" in sample_id:
            df["condition"] = "ctr"
        elif "hst" in sample_id:
            df["condition"] = "hst"
        elif "ftl" in sample_id:
            df["condition"] = "ftl"
        df_flow.append(df)
        
    df_flow = pd.concat(df_flow)
    df_flow.columns = [pns if pns != '' else pnn for pnn, pns in df_flow.columns]
    print('Parameters:', df_flow.keys())
    return df_flow, sample_list, session

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
        A DataFrame containing count data for the flow cytometry samples.

    Returns:
    --------
    anndata.AnnData
        An AnnData object containing the flow cytometry count data with associated metadata.
        The object includes:
        - X: The count matrix from df_flow_counts
        - obs: Observation annotations including 'group' and 'sample_id'

    """
    df_flow['sample_id'] = df_flow['sample_id'].apply(lambda x: x[:4])
    list_metadata = {'group' : df_flow.condition, 'sample_id' : df_flow.sample_id}
    df_metadata = pd.DataFrame(list_metadata)
    
    adata = sc.AnnData(df_flow_counts)
    df_metadata.index = adata.obs.index
    adata.obs['group'] = df_metadata.group
    adata.obs['sample_id'] = df_metadata.sample_id
    
    return adata

def pca_df(sample, session):
    """
    Creates a DataFrame of PCA data grouped by sample ID with donor group classification.

    This function extracts PCA coordinates from an AnnData object, groups them by sample ID,
    and assigns donor group classifications (ctr, hst, ftl) based on sample ID patterns.
    It filters out samples that don't match the grouped PCA indices.

    Parameters:
    -----------
    sample : anndata.AnnData
        An AnnData object containing PCA coordinates in the obsm['X_pca'] attribute
        and sample IDs in the obs['sample_id'] attribute.
    session : flowkit.Session
        A flowkit Session object containing sample information.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing averaged PCA coordinates grouped by sample ID,
        with an additional 'donor_group' column indicating the classification
        (ctr, hst, or ftl) for each sample.
    """
    sample_list = session.get_sample_ids()
    pca_df = pd.DataFrame(sample.obsm['X_pca'], columns=[f'PC{i+1}' for i in range(sample.obsm['X_pca'].shape[1])])
    pca_df['sample_id'] = sample.obs['sample_id'].values

    grouped_pca = pca_df.groupby('sample_id').mean()

    for sample_id in sample_list:
        sample_id_ = sample_id[:4]
        if sample_id_ not in grouped_pca.index:
            sample_list.remove(sample_id)

    donor_groups = []
    for sample_id in sample_list:
        if "ctr" in sample_id:
            donor_groups.append('ctr')
        elif "hst" in sample_id:
            donor_groups.append('hst')
        elif "ftl" in sample_id:
            donor_groups.append('ftl')
            
    grouped_pca['donor_group'] = donor_groups
    return grouped_pca