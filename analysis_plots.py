# analysis_plots.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_coefficient_heatmap(model, feature_names, class_labels=None):
    """
    Creates a heatmap of logistic regression coefficients.
    For a multinomial logistic regression with 5 classes, model.coef_.shape = (5, n_features).
    Rows are classes, columns are features, so we transpose to get (n_features, 5).
    """
    coefs = model.coef_  # shape (n_classes, n_features)
    n_classes, n_features = coefs.shape
    
    if class_labels is None:
        class_labels = list(range(n_classes))  # e.g. 0..4 or 1..5

    # Transpose so each row is a feature, each column is a class
    coefs_T = coefs.T  # shape = (n_features, n_classes)

    fig, ax = plt.subplots(figsize=(16,8))
    # use diverging colormap
    im = ax.imshow(coefs_T, cmap='RdBu', 
                   vmin=-np.max(np.abs(coefs_T)), 
                   vmax=np.max(np.abs(coefs_T)),
                   aspect='auto')

    # Tick labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names)

    ax.set_xlabel("Class label")
    ax.set_ylabel("Features")
    plt.title("Coefficient Heatmap (Logistic Regression)")

    cbar = plt.colorbar(im)
    cbar.set_label("Coefficient value")

    plt.tight_layout()
    plt.show()

def plot_confusion_mat(model, X, y, class_labels=None):
    """
    Plots a confusion matrix comparing model predictions vs. actual.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    if class_labels is None:
        # label them 1..n
        class_labels = sorted(list(set(y)))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_labels,
                yticklabels=class_labels,
                cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
