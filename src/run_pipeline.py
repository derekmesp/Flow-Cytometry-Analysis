import scanpy as sc
import matplotlib.pyplot as plt
import harmonypy as hm
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clustering_pipeline(adata, sample_name, tissue_type, max_iter_harmony=10, theta=0, resolution=0.5, vmin=0, vmax=3):
    """
    Performs PCA, Harmony integration, UMAP embedding, and Leiden clustering on the provided AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data to be processed.
    sample_name : str
        Name of the sample being processed, used for logging and plot titles.
    tissue_type : str
        Type of tissue being analyzed, used for logging and plot titles.
    max_iter_harmony : int, optional
        Maximum number of iterations for the Harmony integration algorithm. Default is 10.
    theta : float, optional
        Diversity clustering penalty parameter for Harmony. Default is 0.
    resolution : float, optional
        Resolution parameter for Leiden clustering. Default is 0.5.
    vmin : float, optional
        Minimum value for color scaling in UMAP plots. Default is 0.
    vmax : float, optional
        Maximum value for color scaling in UMAP plots. Default is 3.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object after processing, containing PCA, Harmony, UMAP, and Leiden clustering results.
    """
    plt.rcParams.update({'font.size': 14})

    logging.info(f"Performing PCA on {sample_name} ({tissue_type})...")
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pl.pca_variance_ratio(adata, log=False)
    sc.pl.pca_loadings(adata, components='1,2',
                       show=False, save='{0}_{1}_PCA_loadings.png'.format(tissue_type, sample_name))

    logging.info(
        f"Running Harmony integration on {sample_name} ({tissue_type})...")
    harmony_out = hm.run_harmony(
        adata.obsm['X_pca'], adata.obs, 'sample_id', max_iter_harmony=max_iter_harmony, theta=theta)
    adata.obsm['X_pca_harmony'] = harmony_out.Z_corr
    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    sc.tl.umap(adata)
    logging.info(
        f"UMAP embedding completed for {sample_name} ({tissue_type}).")

    markers = list(adata.var_names)
    try:
        sc.pl.umap(adata, color=['group'], cmap='turbo',
                   title='{} {} Groups'.format(tissue_type, sample_name), show=False, save='{0}_{1}_UMAP_groups.png'.format(tissue_type, sample_name))
    except Exception as e:
        logging.error(f"Error occurred while plotting UMAP: {e}")

    sc.tl.leiden(adata, resolution=resolution, flavor='leidenalg')
    logging.info(
        f"Leiden clustering results for {sample_name} ({tissue_type}):")
    logging.info(adata.obs['leiden'].value_counts())

    try:
        sc.pl.umap(adata, color=['leiden'], cmap='turbo',
                   title='{} {} Clusters'.format(tissue_type, sample_name), show=False, save='{0}_{1}_UMAP_clusters.png'.format(tissue_type, sample_name))
    except Exception as e:
        logging.error(f"Error occurred while plotting UMAP: {e}")

    try:
        sc.pl.umap(adata, color=['tissue'], cmap='turbo',
                   title='{} Tissue Groups'.format(sample_name), show=False, save='{0}_{1}_UMAP_tissue.png'.format(tissue_type, sample_name))
    except Exception as e:
        logging.error(f"Error occurred while plotting UMAP: {e}")

    try:
        sc.pl.umap(adata, color=markers, cmap='turbo', vmin=vmin, vmax=vmax, show=False,
                   save='{0}_{1}_UMAP_markers.png'.format(tissue_type, sample_name))
    except Exception as e:
        logging.error(f"Error occurred while plotting UMAP: {e}")

    sc.tl.dendogram(adata, groupby='leiden')
    sc.pl.dotplot(adata, markers, swap_axes=True, groupby='leiden', title="{} {} Dotplot".format(
        tissue_type, sample_name), cmap='RdBu_r', dendrogram=True, vcenter=0, vmin=-3, vmax=3, show=False, save='{0}_{1}_Dotplot.png'.format(tissue_type, sample_name)
    )

    return adata


def dem_ranked(adata, groups='leiden', method='t-test'):
    """
    Performs differential expression analysis using the specified method and generates a dot plot of the top 3
    differentially expressed markers for each group.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data to be analyzed.
    groups : str, optional
        The name of the observation column in adata.obs that defines the groups for differential expression analysis. Default is 'leiden'.
    method : str, optional
        The method to use for differential expression analysis. Default is 't-test'.

    Returns
    -------
    """
    sc.tl.rank_genes_groups(adata, groups=groups, method=method)
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    celltype = {'celltype': []}
    cluster_to_markers = {}
    for group in groups:
        top_markers = result['names'][group][:3]
        cluster_to_markers[group] = f"{':'.join(top_markers)} ({group})"

    celltype['celltype'] = [cluster_to_markers[leiden]
                            for leiden in adata.obs['leiden']]
    cell_type_series = pd.Series(celltype['celltype'])
    unique_values = cell_type_series.unique()
    logging.info(f"Unique cell types identified: {unique_values}")

    sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=3,  cmap='RdBu_r', vcenter=0, vmin=-3, vmax=3)

    return adata, unique_values
