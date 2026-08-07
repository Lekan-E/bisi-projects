import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_feature_importance(model, x, save_path="feature_importance.png"):
    """
    Plot a bar chart showing the feature importances.

    Args:
        model: A fitted model exposing `feature_importances_` (e.g. RandomForestClassifier).
        x (pandas.DataFrame): The feature matrix the model was trained on, used for column names.
        save_path (str): Where to save the resulting chart.
    """
    fig, ax = plt.subplots()
    sns.barplot(x=model.feature_importances_, y=x.columns, ax=ax)
    ax.set_title("Feature importance chart")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(save_path)

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png", normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix for the given true and predicted labels.

    Args:
        y_true (numpy.ndarray): Array of true labels.
        y_pred (numpy.ndarray): Array of predicted labels.
        classes (list): List of class labels.
        save_path (str): Where to save the resulting chart.
        normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
        title (str, optional): Title for the plot. Default is 'Confusion Matrix'.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=16)
    fig.savefig(save_path)
