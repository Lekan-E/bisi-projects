def predict_clusters(model, X):
    """
    Assign each row in X to a cluster using the trained model.
    """
    return model.predict(X)

def assign_clusters(df, labels):
    """
    Attach predicted cluster labels back onto the original dataframe.
    """
    df = df.copy()
    df['Cluster'] = labels

    return df
