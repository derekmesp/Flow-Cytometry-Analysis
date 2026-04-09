import flowkit as fk 
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

def read_flow(directory, tissue=None):
    """
    Reads flow cytometry data from FCS files in the specified directory.
    
    This function loads flow cytometry data from FCS files, processes them into a pandas DataFrame,
    and assigns tissue and condition labels based on sample IDs. It automatically detects control
    (ctr), history (hst), and fatal (ftl) asthmatic conditions from the sample names.
    
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
        df['age'] = int(sample_id.split('_')[2])
        df["sample_id"] = sample_id
        if "ctr" in sample_id:
            df["condition"] = "ctr"
            df["asthma"] = "control"
        elif "hst" in sample_id:
            df["condition"] = "hst"
            df["asthma"] = "asthmatic"
        elif "ftl" in sample_id:
            df["condition"] = "ftl"
            df["asthma"] = "asthmatic"
        else:
            manual_condition = manual_condition = input(
                f"Sample {sample_id} missing condition annotation. Insert manually: "
            )
            df["condition"] = manual_condition
            df["asthma"] = "asthmatic" if manual_condition in ["hst", "ftl"] else "control"
            
        df_flow.append(df)
     
    df_flow = pd.concat(df_flow)
    df_flow.columns = [pns if pns != '' else pnn for pnn, pns in df_flow.columns]
    print('Parameters:', df_flow.keys())
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
        matches = [(s2, SequenceMatcher(None, s1, s2).ratio()) for s2 in remaining]
        
        if matches:
            best_match, score = max(matches, key=lambda x: x[1])
            
            if score >= threshold:
                df[s1] = df[[s1, best_match]].max(axis=1)
                print(f"Merged {best_match} into {s1} (Score: {score:.2f})")
                
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
    if df_flow.isna().any().any():
        nan_cols = df_flow.columns[df_flow.isna().any()].tolist()
        df_flow = automate_merging(df_flow, nan_cols)
        
    df_flow['sample_id'] = df_flow['sample_id'].apply(lambda x: x[:4])
    list_metadata = {
        'group': df_flow.condition,
        'sample_id': df_flow.sample_id,
        'age': df_flow.age,
        'asthma': df_flow.asthma
    }

    if 'tissue' in df_flow.columns:
        list_metadata['tissue'] = df_flow.tissue

    df_metadata = pd.DataFrame(list_metadata)
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
        sample_dict[sample_id[:4]] = 'ctr' if 'ctr' in sample_id else 'hst' if 'hst' in sample_id else 'ftl'
    
    pca_df = pd.DataFrame(sample.obsm['X_pca'], columns=[f'PC{i+1}' for i in range(sample.obsm['X_pca'].shape[1])])
    pca_df['sample_id'] = sample.obs['sample_id'].values
    
    grouped_pca = pca_df.groupby('sample_id').mean()
    grouped_pca['donor_group'] = grouped_pca.index.map(sample_dict)
    
    return grouped_pca
