import matplotlib.pyplot as plt
import seaborn as sns

def plot_elbow(wss_df, save_path="elbow_plot.png"):
    """
    Plot the Elbow (WCSS) curve used to pick the number of clusters.
    """
    fig, ax = plt.subplots()
    ax.plot(wss_df['cluster'], wss_df['WCSS_Score'], marker='o')
    ax.set_xlabel('No. of clusters')
    ax.set_ylabel('WCSS Score')
    ax.set_title('Elbow Plot')
    ax.grid(True)
    fig.savefig(save_path)

def plot_silhouette(sil_df, save_path="silhouette_plot.png"):
    """
    Plot the Silhouette score curve used to pick the number of clusters.
    """
    fig, ax = plt.subplots()
    ax.plot(sil_df['cluster'], sil_df['Silhouette_Score'], marker='o')
    ax.set_xlabel('No. of clusters')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Plot')
    ax.grid(True)
    fig.savefig(save_path)

def plot_clusters(df, centers, x='Annual_Income', y='Spending_Score', save_path="cluster_plot.png"):
    """
    Plot the customer segments with their cluster centers.

    Args:
        df (pandas.DataFrame): Customer data with a 'Cluster' column, in original units.
        centers (array-like): Cluster centers as (n_clusters, 2) in the same original
            units as `x`/`y`, e.g. after inverse-transforming a scaled model's centers
            and selecting the two plotted columns.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, hue='Cluster', data=df, palette='colorblind', ax=ax)
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5, marker='X', label='Centroids')
    ax.set_title('Customer Segments')
    ax.legend()
    fig.savefig(save_path)
