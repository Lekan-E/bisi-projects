import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, x, save_path="feature_importance.png"):
    """
    Plot a bar chart showing the trained model's feature importances.

    Args:
        model: A fitted model exposing `feature_importances_` (e.g. RandomForestRegressor).
        x (pandas.DataFrame): The feature matrix the model was trained on, used for column names.
        save_path (str): Where to save the resulting chart.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=model.feature_importances_, y=x.columns, ax=ax)
    ax.set_title("Feature Importance Chart")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(save_path)
