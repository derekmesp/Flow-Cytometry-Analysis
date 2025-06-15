import flowkit as fk 
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
def read_flow(directory, tissue=None):
    """
    Reads flow cytometry data from FCS files in the specified directory.
    
    This function loads flow cytometry data from FCS files, processes them into a pandas DataFrame,
    and assigns tissue and condition labels based on sample IDs. It automatically detects control
    (ctr), heat-stressed (hst), and fetal (ftl) conditions from the sample names.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing FCS files to be analyzed.
    tissue : str, optional
        Tissue type label to assign to all samples. If None, no tissue label is assigned.
        
    Returns:
    --------
    tuple
        A tuple containing three elements:
        - df_flow (pandas.DataFrame): Combined DataFrame of all flow cytometry samples with added
          metadata columns (tissue, sample_id, condition).
        - sample_list (list): List of all sample IDs found in the directory.
        - session (flowkit.Session): The flowkit Session object used to read the FCS files.
    """
    session = fk.Session(fcs_samples=directory)
    sample_list = session.get_sample_ids()
    
    df_flow = []
    for sample_id in sample_list:
        df = session.get_gate_events(sample_id)
        if tissue is not None:
            df['tissue'] = tissue
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
    if 'tissue' not in df_flow.columns:
        list_metadata = {'group' : df_flow.condition, 'sample_id' : df_flow.sample_id}
        df_metadata = pd.DataFrame(list_metadata)
        
        adata = sc.AnnData(df_flow_counts)
        df_metadata.index = adata.obs.index
        adata.obs['group'] = df_metadata.group
        adata.obs['sample_id'] = df_metadata.sample_id
    else:
        list_metadata = {'group' : df_flow.condition, 'sample_id' : df_flow.sample_id, 'tissue' : df_flow.tissue}
        df_metadata = pd.DataFrame(list_metadata)
        
        adata = sc.AnnData(df_flow_counts)
        df_metadata.index = adata.obs.index
        adata.obs['group'] = df_metadata.group
        adata.obs['sample_id'] = df_metadata.sample_id
        adata.obs['tissue'] = df_metadata.tissue
        
    
    return adata
def pca_df(sample, session=None, singular=True, sample_list=None):
    """
    Creates a DataFrame of PCA data from flow cytometry samples and groups it by sample ID.

    This function extracts PCA coordinates from an AnnData object, groups them by sample ID,
    and assigns donor group labels based on sample naming conventions. It handles both single
    session processing and pre-defined sample lists.

    Parameters:
    -----------
    sample : anndata.AnnData
        An AnnData object containing PCA results in the obsm['X_pca'] slot and sample IDs
        in the obs['sample_id'] column.
    session : flowkit.Session, optional
        A flowkit Session object used to retrieve sample IDs when singular=True.
    singular : bool, default=True
        If True (samples are from one tissue), sample IDs are retrieved from the session. If False (multiple tissues), the provided sample_list is used.
    sample_list : list, optional
        A list of sample IDs to process. Required when singular=False (multiple tissues are present).

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing averaged PCA coordinates for each sample ID, with an additional
        'donor_group' column indicating the experimental condition ('ctr', 'hst', or 'ftl')
        based on the sample ID.
    """
    
    if singular:
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