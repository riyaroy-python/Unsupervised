from sklearn.cluster import KMeans
import logging

logging.basicConfig(filename='logs/clustering.log', level=logging.INFO)


def perform_kmeans(data, n_clusters=5):
    """Apply KMeans clustering and return fitted model and labels."""
    try:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)
        logging.info("KMeans clustering successful with %d clusters", n_clusters)
        return model, labels
    except Exception as e:
        logging.error("Error in KMeans clustering: %s", e)
        raise