import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import logging

logging.basicConfig(filename='logs/clustering.log', level=logging.INFO)


def plot_clusters(data, labels):
    """Visualize clusters using PCA dimensionality reduction."""
    try:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)
        df_plot = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
        df_plot['Cluster'] = labels

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=100)
        plt.title('Customer Segments by KMeans')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.savefig("clusters_plot.png")
        logging.info("Cluster plot saved successfully.")
    except Exception as e:
        logging.error("Error plotting clusters: %s", e)
        raise
