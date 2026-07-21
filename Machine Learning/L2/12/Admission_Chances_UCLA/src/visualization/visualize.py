import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix as _confusion_matrix

def plot_loss_curve(model, save_path="loss_curve.png"):
    """
    Plot the training loss curve of the fitted MLPClassifier.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(model.loss_curve_, label='Loss', color='blue')
    ax.set_title('Loss Curve')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    fig.savefig(save_path)

def plot_confusion_matrix(y_true, y_pred, classes, save_path="confusion_matrix.png", normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix for the given true and predicted labels.
    """
    cm = _confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=16)
    fig.savefig(save_path)
