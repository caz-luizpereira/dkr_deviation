import pandas as pd
import numpy as np
import modeling_evaluation as model_eval
from typing import List, Dict, Optional, Any
from sklearn.metrics import ConfusionMatrixDisplay, auc

# Import Plot Libraries
import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curve(
    y_test: pd.DataFrame,
    probability: Any,
    algorithm: str,
    iteration: int,
    location: Optional[str] = None,
) -> None:
    """
    Function to plot the ROC curve and save it in a specific provided location.


    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param probability: The predicted labels probabilities.
    :type probability: Any

    :param algorithm: The name of the ML algorithm related to the ROC curve plot.
    :type algorithm: str

    :param iteration: The number of RW iteration.
    :type iteration: int

    :param location: The location to save the ROC curve plot. By default is None
    :type location: Optional[str]


    :return: None.
    :rtype: None
    """

    if location is None:
        location = 'static/images'

    results = model_eval.get_roc_curve_metrics(y_test, probability)
    fpr = results['fpr']
    tpr = results['tpr']
    roc_auc = results['roc_auc']

    fig, ax = plt.subplots(1, 1)

    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label='ROC curve (area = %0.4f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    fig.savefig(
        '{0}/{1}/ROC_{1}_{2}.pdf'.format(location, algorithm, iteration))


def plot_feature_importance(
    importance: List,
    names: List,
    model_type: str,
    iteration: int,
    location: Optional[str] = None
) -> None:
    """
    Function to plot the feature importance and save it in a specific provided location.


    :param importance: The feature importance array.
    :type importance: List

    :param names: The features name.
    :type names: List

    :param model_type: The name of the ML algorithm.
    :type model_type: str

    :param iteration: The number of RW iteration.
    :type iteration: int

    :param location: The location to save the feature importance plot. By default is None
    :type location: Optional[str]


    :return: None.
    :rtype: None
    """

    if location is None:
        location = 'static/images'

    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names,
            'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.savefig('{}/FI_{}_{}.pdf'.format(location, model_type, iteration))


def roc_curve_all(
    metrics: List[Dict],
    location: Optional[str] = None
) -> None:
    """
    Function to plot the ROC curve of all tested models. Afterwards, the generated plot
    is saved in a specific provided location.


    :param metrics: The metrics used to plot the ROC curve of each tested models. Each position of the List contains a
                    Dictionary with the model name and fpr, tpr, and auc metrics.
    :type metrics: List[Dict]

    :param location: The location to save the ROC curve plot. By default is None
    :type location: Optional[str]


    :return: None.
    :rtype: None
    """

    fig, ax = plt.subplots(1, 1)

    if location is None:
        location = 'static/images'

    configs = [
        {'color': 'tab:red', 'linestyle': 'solid'},
        {'color': 'tab:blue', 'linestyle': 'dotted'},
        {'color': 'tab:orange', 'linestyle': 'dashed'},
        {'color': 'black', 'linestyle': '0, (3, 5, 1, 5, 1, 5)'},
        {'color': 'tab:purple', 'linestyle': '0, (5, 10)'},
        {'color': 'tab:green', 'linestyle': 'dotted'},
        {'color': 'tab:gray', 'linestyle': 'solid'}
    ]

    for i, v in enumerate(metrics):
        ax.plot(
            metrics[i]['fpr'],
            metrics[i]['tpr'],
            color=configs[i]['color'],
            linestyle=configs[i]['linestyle'],
            lw=2,
            label='{} (AUC = {:.4f})'.format(metrics[i]['auc'], metrics[i]['model']))

    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(b=True, linestyle='dashed')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    fig.savefig('{}/ALL_ROC.pdf'.format(location))


def plot_confusion_matrix(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame,
    model_name: str,
    iteration: int,
    location: Optional[str] = None,
) -> None:
    """
    Function to plot the confusion matrix from model predictions.


    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param y_pred: The predicted labels.
    :type y_pred: pd.DataFrame

    :param model_name: The name of the ML model.
    :type model_name: str

    :param iteration: The number of RW iteration.
    :type iteration: int

    :param location: The location to save the confusion matrix plot. By default is None
    :type location: Optional[str] = None


    :return: None.
    :rtype: None


    .. seealso:: `Confusion Matrix visualization  \
        <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html>`_

        1. `Save figure \
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>`_
    """

    if location is None:
        location = 'static/images'

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.savefig('{}/CM_{}_{}'.format(location, model_name, iteration))
