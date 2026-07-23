import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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
