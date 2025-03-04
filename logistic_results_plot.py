# analysis_plots.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd


##################################
### FEATURE SIGNIFICANCE PLOTS ###
##################################

def plot_feature_significance(results_df, significance_threshold=0.05):

    plt.figure(figsize=(10, 6))
    
    # Scatter plot: x = P-value, y = abs_coef
    plt.scatter(results_df["P-value"], results_df["abs_coef"], alpha=0.7)
    
    # Vertical line at significance threshold
    plt.axvline(
        x=significance_threshold, 
        color="red", 
        linestyle="--", 
        label=f"Significance Threshold ({significance_threshold})"
    )
    
    plt.xlabel("P-value")
    plt.ylabel("Absolute Coefficient")
    plt.title("Feature Significance vs. Effect Size")
    plt.legend()
    plt.show()

def plot_coefficients_barh_by_abscoef(results_df):
    """
    Sorts the features by absolute coefficient in ascending order
    and plots a horizontal bar chart of raw coefficients.
    Positive coefficients in green, negative in red.
    """
    # Sort
    results_df_sorted = results_df.sort_values(by="abs_coef", ascending=True)
    
    plt.figure(figsize=(10, 16))
    
    # Color: green if positive, red if negative
    colors = ["green" if x > 0 else "red" for x in results_df_sorted["Coefficient"]]
    
    plt.barh(
        results_df_sorted["Feature"], 
        results_df_sorted["Coefficient"], 
        color=colors
    )
    plt.axvline(0, color="black", linewidth=1)  # vertical line at 0
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Feature Influence on V2G Adoption (Logistic Regression)")
    plt.show()

def plot_odds_ratio_barh(results_df):
    """
    Sorts the features by odds ratio (descending) and plots 
    a horizontal bar chart of odds ratios with a reference line at x=1.
    """
    results_df_sorted = results_df.sort_values(by="Odds Ratio", ascending=False)

    plt.figure(figsize=(16, 10))
    sns.barplot(
        data=results_df_sorted,
        x='Odds Ratio',
        y='Feature',
        orient='h',
        palette='crest'
    )
    # Reference line at odds ratio=1
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Odds Ratio')
    plt.ylabel('Features')
    plt.title('Odds Ratios for Logistic Regression Features')
    plt.tight_layout()
    plt.show()


def plot_p_value_barh(results_df):
    """
    Sorts by p-value ascending, then plots a horizontal bar chart of p-values.
    Colors bars blue if p<0.05, else gray. Reference line at p=0.05.
    """
    results_df_sorted = results_df.sort_values(by="P-value", ascending=True)
    
    plt.figure(figsize=(10, 12))
    colors = ["blue" if p < 0.05 else "gray" for p in results_df_sorted["P-value"]]
    
    plt.barh(results_df_sorted["Feature"], results_df_sorted["P-value"], color=colors)
    plt.axvline(0.05, color="red", linestyle="dashed", linewidth=1.5, label="Significance Threshold (p=0.05)")
    plt.xlabel("P-Value")
    plt.ylabel("Feature")
    plt.title("Feature Significance (P-Values)")
    plt.legend()
    plt.show()


def plot_log_p_value_barh(results_df):
    """
    Similar to plot_p_value_barh, but uses log10(p-value).
    Sort by p-value ascending, bar chart of log10(p-value).
    Blue if p<0.05, else gray. Reference line at np.log10(0.05).
    """
    results_df_sorted = results_df.sort_values(by="P-value", ascending=True)
    
    plt.figure(figsize=(10, 12))
    colors = ["blue" if p < 0.05 else "gray" for p in results_df_sorted["P-value"]]
    log_pvals = np.log10(results_df_sorted["P-value"])
    
    plt.barh(results_df_sorted["Feature"], log_pvals, color=colors)
    plt.axvline(np.log10(0.05), color="red", linestyle="dashed", linewidth=1.5, 
                label="Significance Threshold (log10 p=0.05)")
    plt.xlabel("Log10 P-Value")
    plt.ylabel("Feature")
    plt.title("Feature Significance (Log-Scaled P-Values)")
    plt.legend()
    plt.show()


def plot_coefficients_significance_barh(results_df):
    """
    Sort by abs_coef ascending. 
    Features with p>=0.05 => gray, positive => green, negative => red.
    Plots horizontal bar chart, labeling p-values on the right side of each bar.
    """
    results_df_sorted = results_df.sort_values(by="abs_coef", ascending=True)

    # Color logic: 
    #  - if p>=0.05 => gray
    #  - else positive => green, negative => red
    colors = [
        "gray" if p >= 0.05 else ("green" if coef > 0 else "red")
        for p, coef in zip(results_df_sorted["P-value"], results_df_sorted["Coefficient"])
    ]

    plt.figure(figsize=(12, 18))
    bars = plt.barh(results_df_sorted["Feature"], results_df_sorted["Coefficient"], color=colors)

    # Label p-values to the right of each bar
    for bar, p_value in zip(bars, results_df_sorted["P-value"]):
        # Position text at the right edge + a small offset
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                 f"p={p_value:.3g}", va='center', fontsize=10)

    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Feature Influence on V2G Adoption (Logistic Regression)\n(Non-significant features greyed out)")
    plt.tight_layout()
    plt.show()

def plot_coefficient_vs_significance(results_df, p_threshold=0.05):
    """
    Plots a scatter of 'Coefficient' vs. '-log10(P-value)' from your results_df.
    
    Marks points where p < p_threshold in one color (e.g. blue) 
    and non-significant in another (e.g. gray). 
    Adds reference lines at x=0 (no effect) and y=-log10(p_threshold).
    
    :param results_df: DataFrame with at least "Coefficient", "P-value", and "Feature".
    :param p_threshold: The significance threshold for coloring points / reference line (default=0.05).
    """
    # 1) Compute -log10(P-value) if not already in the DataFrame
    if "-log(p)" not in results_df.columns:
        results_df["-log(p)"] = -np.log10(results_df["P-value"])

    # 2) Define colors based on significance threshold
    colors = ["blue" if p < p_threshold else "gray" for p in results_df["P-value"]]

    # 3) Create the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        results_df["Coefficient"], 
        results_df["-log(p)"], 
        c=colors, 
        edgecolors="black"
    )

    # 4) Annotate each point with the Feature name
    for i, txt in enumerate(results_df["Feature"]):
        plt.annotate(
            txt, 
            (results_df["Coefficient"].iloc[i], results_df["-log(p)"].iloc[i]), 
            fontsize=9, 
            alpha=0.7
        )

    # 5) Reference lines
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)  # Vertical line at zero coefficient
    plt.axhline(
        y=-np.log10(p_threshold), 
        color="red", 
        linestyle="--", 
        linewidth=1, 
        label=f"p={p_threshold} threshold"
    )

    # 6) Labels & title
    plt.xlabel("Coefficient")
    plt.ylabel("-log10(P-value)")
    plt.title("Feature Influence on V2G Adoption (Coefficient vs Significance)")
    plt.legend()
    plt.show()

#################################
### LOGISTIC REGRESSION PLOTS ###
#################################

def plot_coefficients_barplot(coefs_df, feature_col="Feature", coef_col="Coefficient", title="Effect size in Predicting V2G Adoption (Binary Logistic Regression)"):
    """
    Plots a barplot of coefficients (log-odds) from a logistic regression.
    
    :param coefs_df: DataFrame with columns:
        - feature_col (e.g. "Feature") for label
        - coef_col (e.g. "Coefficient") for coefficient
    :param feature_col: name of the column containing feature labels
    :param coef_col: name of the column containing coefficient values
    :param title: title for the plot
    """
    # Sort descending by coefficient
    plot_df = coefs_df.sort_values(by=coef_col, ascending=False).copy()

    plt.figure(figsize=(12, 8))
    sns.barplot(x=coef_col, y=feature_col, data=plot_df, palette="coolwarm")
    plt.title(title)
    plt.xlabel("Coefficient Value (Log Odds)")
    plt.ylabel("Feature")
    plt.axvline(x=0, color='black', linestyle='--')  # reference line for neutral impact
    plt.show()

def plot_odds_ratios_barplot(coefs_df, feature_col="Feature", coef_col="Coefficient", title="Effect size in Predicting V2G Adoption (Binary Logistic Regression)"):
    """
    Plots a barplot of odds ratios from a logistic regression.
    
    :param coefs_df: DataFrame with columns:
        - feature_col (e.g. "Feature") for label
        - coef_col (e.g. "Coefficient") for the log-odds coefficient
        (Assumes you want to compute 'Odds Ratio' = exp(coefficient).)
    :param feature_col: name of the column containing feature labels
    :param coef_col: name of the column containing coefficient values
    :param title: title for the plot
    """
    plot_df = coefs_df.copy()
    # Compute odds ratio
    plot_df["Odds Ratio"] = np.exp(plot_df[coef_col])
    # Sort descending by odds ratio
    plot_df.sort_values(by="Odds Ratio", ascending=False, inplace=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Odds Ratio", y=feature_col, data=plot_df, palette="coolwarm")
    plt.axvline(x=1, color='black', linestyle='--')  # reference line for OR=1
    plt.title(title)
    plt.xlabel("Odds Ratio")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def plot_coefficients_by_abs(coefs_df, feature_col="Feature", coef_col="Coefficient", title="Feature Importance in Predicting V2G Adoption (Binary Logistic Regression)"):
    """
    Plots a barplot of coefficients sorted by absolute value (magnitude).
    
    :param coefs_df: DataFrame with at least:
        - feature_col (e.g. "Feature")
        - coef_col (e.g. "Coefficient")
    :param feature_col: name of column containing feature labels
    :param coef_col: name of column containing coefficient values
    :param title: title for the plot
    """
    plot_df = coefs_df.copy()
    plot_df["abs_coef"] = plot_df[coef_col].abs()
    plot_df.sort_values(by="abs_coef", ascending=False, inplace=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=coef_col, y=feature_col, data=plot_df, palette="coolwarm")
    plt.title(title)
    plt.xlabel("Coefficient Value (Log Odds)")
    plt.ylabel("Feature")
    plt.axvline(x=0, color='black', linestyle='--')  # reference line for neutral impact
    plt.show()

def plot_binary_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix: Binary V2G Adoption Prediction"):
    """
    Plots a confusion matrix using sklearn's ConfusionMatrixDisplay.
    
    :param y_true: array-like of shape (n_samples,), ground truth labels
    :param y_pred: array-like of shape (n_samples,), predicted labels
    :param labels: list of label names in the order to be displayed, e.g. ["Not Adopting", "Adopting"]
    :param title: title for the plot
    """
    if labels is None:
        # default for 0 => "Negative", 1 => "Positive"
        labels = ["Label=0", "Label=1"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

#############
### OTHER ###
#############

def transform_and_plot_correlation(
    df, 
    transformers, 
    figsize=(20, 16), 
    title="Correlation Matrix of Transformed Features"
):
    """
    Applies transformers to each column of the given DataFrame (if a transformer exists),
    converts columns to numeric, REPLACES -1 with NaN, computes the correlation matrix,
    and plots a heatmap.
    
    :param df: The input pandas DataFrame to be transformed and analyzed.
    :param transformers: A dictionary mapping column_name -> transformer_function,
                         where each transformer_function takes a raw cell value and
                         returns a numeric (or -1 if invalid).
    :param figsize: Tuple specifying figure size for the heatmap, default (20,16).
    :param title: Title for the correlation heatmap.
    
    :return: The correlation matrix (as a pandas DataFrame).
    """

    # 1) Make a copy of the df to avoid mutating the original
    transformed_df = df.copy()

    # 2) Apply transformers where available
    for col in transformed_df.columns:
        if col in transformers:
            transform_func = transformers[col]
            transformed_df[col] = transformed_df[col].apply(transform_func)
            # Replace -1 with NaN to avoid skewing correlation for invalid entries
            transformed_df[col] = transformed_df[col].replace(-1, np.nan)

    # 3) Ensure numeric values
    #    (strings -> float, or invalid -> NaN if parse fails)
    transformed_df = transformed_df.apply(pd.to_numeric, errors='coerce')

    # 4) Compute correlation matrix
    corr_matrix = transformed_df.corr()

    # 5) Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix, 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f", 
        vmin=-1, 
        vmax=1
    )
    plt.title(title)
    plt.show()
    
    # 6) Return the correlation matrix
    return corr_matrix
    

def plot_coefficient_heatmap(model, feature_names, class_labels=None):
    """
    Creates a heatmap of logistic regression coefficients.
    For a multinomial logistic regression with 5 classes, model.coef_.shape = (5, n_features).
    Rows are classes, columns are features, so we transpose to get (n_features, 5).
    """
    coefs = model.coef_  # shape (n_classes, n_features)
    n_classes, n_features = coefs.shape
    
    if class_labels is None:
        class_labels = list(range(n_classes))

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
