import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def scale_features(X):
    """
    Standardize features so no single feature (e.g. Annual_Income's wider raw range)
    dominates the distance calculation KMeans relies on.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler

def elbow_method(X, k_range=range(3, 9)):
    """
    Compute the within-cluster sum of squares (WCSS) for a range of k values.
    """
    K, WCSS = [], []
    for k in k_range:
        kmodel = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)
        K.append(k)
        WCSS.append(kmodel.inertia_)

    return pd.DataFrame({'cluster': K, 'WCSS_Score': WCSS})

def silhouette_method(X, k_range=range(3, 9)):
    """
    Compute the silhouette score for a range of k values.
    """
    K, scores = [], []
    for k in k_range:
        kmodel = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)
        K.append(k)
        scores.append(silhouette_score(X, kmodel.labels_))

    return pd.DataFrame({'cluster': K, 'Silhouette_Score': scores})

def train_kmeans(X, n_clusters, scaler):
    """
    Train the final KMeans model and persist it, along with the scaler used to
    prepare its input, to disk.
    """
    kmodel = KMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=42).fit(X)

    with open('models/KMeansmodel.pkl', 'wb') as file:
        pickle.dump(kmodel, file)

    with open('models/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    return kmodel
