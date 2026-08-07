def select_features(df, feature_cols=['Age', 'Annual_Income', 'Spending_Score']):
    """
    Select the features the clustering model is trained on.

    Args:
        df (pandas.DataFrame): The preprocessed customer data.
        feature_cols (list): Columns to use for clustering.
    """
    X = df[feature_cols]

    return X
