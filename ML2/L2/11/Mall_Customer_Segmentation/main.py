import pandas as pd
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import select_features
from src.models.train_model import scale_features, elbow_method, silhouette_method, train_kmeans
from src.models.predict_model import predict_clusters, assign_clusters
from src.visualization.visualize import plot_elbow, plot_silhouette, plot_clusters

from logger import get_logger
logger = get_logger(__name__)

FEATURE_COLS = ['Age', 'Annual_Income', 'Spending_Score']

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline.")

        # Load and preprocess the data
        data_path = "data/raw/mall_customers.csv"
        df = load_and_preprocess_data(data_path)
        df.to_csv("data/processed/Processed_Mall_Customers.csv", index=False)
        logger.info("Loaded and preprocessed %d rows from %s.", len(df), data_path)

        # Select the features the clustering model is trained on
        X = select_features(df, FEATURE_COLS)

        # Age, Annual_Income, and Spending_Score live on different scales,
        # so standardize them before computing distances
        X_scaled, scaler = scale_features(X)

        # Determine the optimal number of clusters via Elbow and Silhouette methods
        wss = elbow_method(X_scaled)
        sil = silhouette_method(X_scaled)
        plot_elbow(wss)
        plot_silhouette(sil)
        logger.info("WCSS scores:\n%s", wss)
        logger.info("Silhouette scores:\n%s", sil)

        #best_k = int(sil.loc[sil['Silhouette_Score'].idxmax(), 'cluster'])
        best_k = 5
        logger.info("Using k=%d clusters.", best_k)
        print(f"Using k={best_k} clusters.\n")

        model = train_kmeans(X_scaled, n_clusters=best_k, scaler=scaler)
        logger.info("Model training complete.")

        # Assign every customer to a segment and visualize the result
        labels = predict_clusters(model, X_scaled)
        df = assign_clusters(df, labels)

        # Cluster centers live in scaled space; bring them back to original units so
        # they land on the same axes as the plotted (unscaled) Income/Spending data
        centers_orig = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURE_COLS)
        plot_clusters(df, centers_orig[['Annual_Income', 'Spending_Score']].values)

        cluster_sizes = df['Cluster'].value_counts()
        cluster_profile = df.groupby('Cluster')[['Age', 'Annual_Income', 'Spending_Score']].mean()
        logger.info("Cluster sizes:\n%s", cluster_sizes)
        logger.info("Cluster profile (mean values):\n%s", cluster_profile)

        print(f"Cluster sizes:\n{cluster_sizes}\n")
        print(f"Cluster profile (mean values):\n{cluster_profile}")
    except Exception:
        logger.exception("Training pipeline failed.")
        raise
