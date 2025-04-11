from modules.data_loader import load_data, preprocess_data
from modules.model import perform_kmeans
from modules.utils import plot_clusters

if __name__ == '__main__':
    df = load_data("data/raw/mall_customers.csv")
    processed = preprocess_data(df)
    model, labels = perform_kmeans(processed)
    plot_clusters(processed, labels)
    print("Clustering complete. Plot saved as clusters_plot.png")