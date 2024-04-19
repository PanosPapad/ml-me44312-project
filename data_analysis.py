from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import shap
from shap import Explanation
from sklearn.metrics import ConfusionMatrixDisplay


def class_to_abbr(c: str) -> str:
    """
    Abbreviates the classification choices for use in image saving.

    :param c: The classification choice to abbreviate.
    :return:  The abbreviation of the given classification choice.
    """
    match c:
        case 'Public Transports':
            abbr = 'PT'
        case 'Private Modes':
            abbr = 'PM'
        case 'Soft Modes':
            abbr = 'SM'
        case _:
            abbr = ''

    return abbr


def save_fig(file: str, save: bool) -> None:
    """
    Saves the current figure to the file as a png in images/.

    :param file: The file name to save the figure to.
    :param save: Boolean denoting if the current figure should be saved.
    :return:     None.
    """
    if save:
        fig = plt.gcf()
        fig.savefig('images/' + file + '.png')


def data_correlation(file: str, threshold: float, save: bool = False) -> None:
    """
    Creates and shows a correlation matrix of all the data in the file.
    It also prints any correlated columns with a correlation above the specified threshold value.

    :param file:      String of a CSV data file for which the correlation matrix should be made.
    :param threshold: The correlation threshold between 0 and 1 for which columns should be printed.
    :param save:      Boolean denoting if the resulting image should be saved.
    :return:          None.
    """
    filtered_data = pd.read_csv(file)

    # Calculate the correlation matrix
    corr = filtered_data.corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(40, 40))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Features")
    save_fig('correlation_heatmap', save)
    plt.show()

    # Extract highly correlated features
    highly_correlated_features = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                colname_i = corr.columns[i]
                colname_j = corr.columns[j]
                highly_correlated_features.add(colname_i)
                highly_correlated_features.add(colname_j)

    print("Highly correlated features (>", threshold, "):", sorted(highly_correlated_features))


def pretty_plot(y_label: str, y_font: int, title: str, title_font: int, title_x: float,
                title_y: float, file: str, save: bool) -> None:
    """
    Adds a y label and title to a plot, and plots tightly.

    :param y_label:    String representing the y label.
    :param y_font:     Integer font size of the y label.
    :param title:      String representing the title.
    :param title_font: Integer font size of the title.
    :param title_x:    Horizontal offset between 0 and 1 of the title placement.
    :param title_y:    Vertical offset between 0 and 1 of the title placement.
    :param file:
    :param save:       Boolean denoting if the resulting image should be saved.
    :return:           None.
    """
    # Get current plot axis
    ax = plt.gca()

    ax.set_ylabel(y_label, fontsize=y_font)
    plt.title(title, fontsize=title_font, x=title_x, y=title_y)

    plt.tight_layout()
    save_fig(file, save)
    plt.show()


def shap_plot(shap_val: Explanation, class_names: list[str], plot_type: str, save: bool = False) -> None:
    """
    Plots SHAP values in a specified plot type.

    :param shap_val:    SHAP explainer object, commonly referred to as SHAP values (also includes base values and data).
    :param class_names: List of strings representing the names of the classification choices.
    :param plot_type:   String representing which SHAP plot should be used.
    :param save:        Boolean denoting if the resulting images should be saved.
    :return:            None.
    """
    # For each classification choice, show a plot
    for i, c in enumerate(class_names):
        # Set shared labels and title
        ylabel = 'Features'
        title = 'Mean SHAP values for ' + c
        # Plot the requested plot type
        match plot_type:
            case 'bar':
                shap.plots.bar(shap_val[:, :, i], show=False)
            case 'beeswarm':
                shap.plots.beeswarm(shap_val[:, :, i], show=False)
                title = 'SHAP values for ' + c
            case 'scatter':
                # Generate SHAP values for each feature
                df = pd.DataFrame(columns=['SHAP_value', 'Feature'])
                for f in range(len(shap_val.feature_names)):
                    df.loc[len(df)] = [np.mean(np.absolute(shap_val[:, f, i].values)), shap_val.feature_names[f]]

                # Sort SHAP values such that the most important one is at the top
                df = df.sort_values(by=['SHAP_value'], ascending=False)

                # Take the nine most important features
                imp_feat = df['Feature'].tolist()[:9]

                # Plot the SHAP values of the nine most important features
                shap.plots.scatter(shap_val[:, imp_feat, 0], show=False)
                ylabel = f"SHAP value\n(higher means more likely to take {class_to_abbr(c)})"
                title = ''
            case _:
                shap.plots.bar(shap_val[:, :, i], show=False)

        # File to save to
        file = class_to_abbr(c) + '_SHAP_' + plot_type

        pretty_plot(ylabel, 14, title, 20, 0.25, 1, file=file, save=save)


def confusion_matrices(model: Any, X_test_scaled: pd.DataFrame, y_test: pd.Series, class_names: list[str],
                       save: bool = False, solver: str = 'lbfgs') -> None:
    """
    Plots the normalized and non-normalized confusion matrices of a machine learning model's
    prediction on the test set.

    :param model:         Fitted machine learning model on a train set.
    :param X_test_scaled: Pandas DataFrame containing the scaled features of the test set.
    :param y_test:        Pandas DataFrame containing the true classification labels of the test set.
    :param class_names:   List of strings representing the names of the classification choices.
    :param save:          Boolean denoting if the resulting images should be saved.
    :param solver:        String denoting which solving technology was used in model creation.
    :return:              None.
    """
    # The two matrix titles
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    # Creates confusion and normalized confusion matrices
    for title, normalize in titles_options:
        fig = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test_scaled,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        fig.ax_.set_title(title)

        fig.figure_.set_size_inches(6, 5)
        plt.tight_layout()

        # File to save to
        file = solver + '_confusion_matrix'
        save_fig(file + '_norm' if normalize else file, save)

    plt.show()


def shap_sum_bar_plot(shap_val: Explanation, class_names: list[str]) -> None:
    """
    Plots the mean of the absolute SHAP values of a machine learning model for each
    classification choice in an aggregated bar plot.

    :param shap_val:    SHAP explainer object, commonly referred to as SHAP values (also includes base values and data).
    :param class_names: List of strings representing the names of the classification choices.
    :return:            None.
    """
    # Create new DataFrame for all SHAP values
    df_shap_val = pd.DataFrame(columns=['SHAP_val', 'Feature', 'Choice'])

    # For each classification choice and feature, calculate the average absolute SHAP value.
    for c in range(shap_val.shape[2]):
        for f in range(len(shap_val.feature_names)):
            df_shap_val.loc[len(df_shap_val)] = [np.mean(np.absolute(shap_val[:, f, c].values)),
                                                 shap_val.feature_names[f], class_names[c]]

    # Plot the SHAP values as a stacked bar plot
    fig = px.bar(df_shap_val, x='SHAP_val', y='Feature', orientation='h', color='Choice')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'},
                      xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)',
                      yaxis_title='Features',
                      legend=dict(x=0.99, y=0.8, xanchor='right', yanchor='bottom'))

    fig.show()
