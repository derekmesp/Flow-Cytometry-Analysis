import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def pca_plot(grouped_pca, tissue_type, sample_name):
    """
    Creates a Principal Component Analysis (PCA) scatter plot with samples grouped by donor type.

    This function visualizes PCA results by plotting PC1 vs PC2, with each point representing a sample.
    Samples are color-coded by donor group, labeled with their IDs, and encircled by group-specific
    boundaries. The plot includes a legend identifying each donor group.

    Parameters
    ----------
    grouped_pca : pandas.DataFrame
        DataFrame containing PCA results with columns 'PC1', 'PC2', and 'donor_group'.
        The index of the DataFrame should contain sample identifiers.
    tissue_type : str
        The type of tissue being analyzed, used in the plot title.
    sample_name : str
        The name or identifier of the sample set, used in the plot title.

    Returns
    -------
    None
        The function displays the plot but does not return any value.
    """
    donor_colors = {
        'ctr': 'grey',
        'hst': 'pink',
        'ftl': 'red'
    }

    colors = grouped_pca['donor_group'].map(
        lambda x: donor_colors.get(x, 'black'))

    plt.figure(figsize=(8, 6))
    plt.scatter(grouped_pca['PC1'], grouped_pca['PC2'],
                c=colors, edgecolor='k', s=100)

    for sample_id, (x, y) in grouped_pca[['PC1', 'PC2']].iterrows():
        plt.text(x, y, sample_id, fontsize=9)

    for group, color in donor_colors.items():
        subset = grouped_pca[grouped_pca['donor_group'] == group]
        if not subset.empty:
            centroid_x, centroid_y = subset[['PC1', 'PC2']].mean()
            max_distance = np.max(np.linalg.norm(
                subset[['PC1', 'PC2']].values - np.array([centroid_x, centroid_y]), axis=1))
            circle = plt.Circle((centroid_x, centroid_y), max_distance,
                                color=color, fill=False, linestyle='--', linewidth=2)
            plt.gca().add_patch(circle)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('{} {} PCA'.format(tissue_type, sample_name))

    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10)
               for color in donor_colors.values()]
    plt.legend(handles, donor_colors.keys(), title='Donor Group',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def tSNE_plot(grouped_pca, tissue_type, sample_name):
    """
    Creates a t-Distributed Stochastic Neighbor Embedding (t-SNE) scatter plot with samples grouped by donor type.

    This function applies t-SNE dimensionality reduction to the input data and visualizes the results
    by plotting tSNE1 vs tSNE2, with each point representing a sample. Samples are color-coded by donor 
    group, labeled with their IDs, and encircled by group-specific boundaries. The plot includes a legend 
    identifying each donor group.

    Parameters
    ----------
    grouped_pca : pandas.DataFrame
        DataFrame containing feature data with the last column assumed to be 'donor_group'.
        The index of the DataFrame should contain sample identifiers.
    tissue_type : str
        The type of tissue being analyzed, used in the plot title.
    sample_name : str
        The name or identifier of the sample set, used in the plot title.

    Returns
    -------
    None
        The function displays the plot but does not return any value.
    """
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    tsne_results = tsne.fit_transform(grouped_pca.iloc[:, :-1])

    grouped_pca['tSNE1'] = tsne_results[:, 0]
    grouped_pca['tSNE2'] = tsne_results[:, 1]

    donor_colors = {
        'ctr': 'grey',
        'hst': 'pink',
        'ftl': 'red'
    }

    colors = grouped_pca['donor_group'].map(
        lambda x: donor_colors.get(x, 'black'))

    plt.figure(figsize=(8, 6))
    plt.scatter(grouped_pca['tSNE1'], grouped_pca['tSNE2'],
                c=colors, edgecolor='k', s=100)

    for sample_id, (x, y) in grouped_pca[['tSNE1', 'tSNE2']].iterrows():
        plt.text(x, y, sample_id, fontsize=9)

    for group, color in donor_colors.items():
        subset = grouped_pca[grouped_pca['donor_group'] == group]
        if not subset.empty:
            centroid_x, centroid_y = subset[['tSNE1', 'tSNE2']].mean()
            max_distance = np.max(np.linalg.norm(
                subset[['tSNE1', 'tSNE2']].values - np.array([centroid_x, centroid_y]), axis=1))
            circle = plt.Circle((centroid_x, centroid_y), max_distance,
                                color=color, fill=False, linestyle='--', linewidth=2)
            plt.gca().add_patch(circle)

    plt.xlabel('tSNE1')
    plt.ylabel('tSNE2')
    plt.title('{} {} t-SNE'.format(tissue_type, sample_name))

    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10)
               for color in donor_colors.values()]
    plt.legend(handles, donor_colors.keys(), title='Donor Group',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def annotated_umap(adata, tissue_type, sample_name, obs):
    """
    Creates a UMAP plot with annotated clusters.

    This function visualizes UMAP results from an AnnData object, coloring the points by cell type
    and annotating the clusters with their respective labels. The plot includes a title indicating
    the tissue type and sample name.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing UMAP coordinates in the obsm['X_umap'] attribute and cell type
        information in the specified observation column.
    tissue_type : str
        The type of tissue being analyzed, used in the plot title.
    sample_name : str
        The name or identifier of the sample set, used in the plot title.
    obs : str
        The name of the observation column in adata.obs that contains the cell type information.

    Returns
    -------
    None
        The function displays the UMAP plot but does not return any value.
    """
    sc.pl.umap(
        adata,
        color=[obs],
        cmap='turbo',
        title='{} {} celltypes'.format(tissue_type, sample_name),
        show=False,
    )

    ax = plt.gca()
    for cluster in adata.obs['leiden'].cat.categories:
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_coords = adata.obsm['X_umap'][cluster_mask]
        x, y = cluster_coords[:, 0].mean(), cluster_coords[:, 1].mean()
        ax.text(x, y, cluster, color='black', fontsize=10,
                weight='bold', ha='center', va='center')

    plt.show()


def composition_dotplot(adata, group_x, group_y):
    """
    Creates a dot plot showing the composition of cell types across different groups.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the data to be plotted. The observations (cells) should have
        categorical annotations for both group_x and group_y in adata.obs.
    group_x : str
        The name of the observation column in adata.obs that defines the x-axis groups (e
        g., different conditions or samples).
    group_y : str
        The name of the observation column in adata.obs that defines the y-axis groups (e.g., cell types).

    Returns
    -------
    None
        The function displays the dot plot but does not return any value.
    """

    cell_counts = (
        adata.obs
        .groupby([group_x, group_y])
        .size()
        .reset_index(name='counts')
    )

    cell_counts['fraction'] = (
        cell_counts
        .groupby(group_x)['counts']
        .transform(lambda x: x / x.sum())
    )
    plt.figure(figsize=(7, 5))

    dot_data = cell_counts.copy()
    lineage_order = adata.obs[group_y].unique()

    dot_data[group_y] = pd.Categorical(
        dot_data[group_y],
        categories=lineage_order[::-1],
        ordered=True
    )

    sns.scatterplot(
        data=dot_data,
        x=group_x,
        y=group_y,
        size='fraction',
        hue='fraction',
        sizes=(30, 350),
        palette='viridis',
        edgecolor='black',
        linewidth=0.4,
        legend='auto'
    )

    plt.title('Cluster Composition Across {}'.format(group_x), fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=0)

    plt.legend(
        title='Fraction of cells',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False
    )


plt.tight_layout()
plt.show()
