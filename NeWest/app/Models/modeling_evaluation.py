
import pandas as pd
import numpy as np
from typing import Dict, List, Any, cast
import time

# Bayesian optimization (Hyperparameter tuning)
from hyperopt import STATUS_OK

# Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

# Classification Models
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def number_iterations_rolling_window(
    D: pd.DataFrame,
    W: int,
    T: int,
    S: int
) -> int:
    """
    Function to calculate the number of iterations regarding the Rolling Window (RW) mechanism,
    using the following formula:  U = (D-(W+T))/S


    :param D: The Dataframe of Weighings.
    :type D: pd.DataFrame

    :param W: The training data size.
    :type W: int

    :param T: The test data size.
    :type T: int

    :param S: The sliding (jump/slide at each iteration).
    :type S: int


    :return: Number of RW iterations.
    :rtype: int
    """

    D = len(D)
    U = (D - (W + T)) / S
    return int(U)


def rolling_mechanism(
    df_size: int,
    window: int,
    increment: int,
    iteration: int,
    sliding: int
) -> Dict:
    """
    Function to implement the rolling window mechanism (calculate the start and end of training and test size).


    :param df_size: The Dataframe size.
    :type df_size: int

    :param window: The Dataframe size.
    :type window: int

    :param increment: The test data size
    :type increment: int

    :param iteration: The current rolling window iteraction
    :type iteration: int

    :param sliding: The sliding (jump/slide at each iteration)
    :type sliding: int


    :return: Train and Test set range.
    :rtype: Dict
    """

    # Calculate the start and end of W (training data size)
    end_train = window + increment * (iteration - 1)
    end_train = min(end_train, df_size)
    start_train = max((end_train - window + 1), 1)
    TR = [start_train, end_train]

    # Calculate the start and end of T (test data size)
    end_test = end_train + sliding
    end_test = min(end_test, df_size)
    start_test = end_train + 1
    if start_test < end_test:
        TS = [start_test, end_test]
    else:
        TS = None

    return {"tr": TR, "ts": TS}


def get_train_test_set(
    rolling: Dict,
    X: pd.DataFrame,
    y: pd.DataFrame
) -> Dict:
    """
    Function to get train and test set, as well the corresponding labels.


    :param rolling: The rolling window mechanism (window size for train and test set).
    :type rolling: Dict

    :param X: The training Dataframe.
    :type X: pd.DataFrame

    :param y: The testing Dataframe.
    :type y: pd.DataFrame


    :return: X_train (Training set), y_train (Training labels), X_test (Testing set) and y_test (Testing labels).
    :rtype: Dict


    .. seealso:: `Purely integer-location based indexing for selection by position  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html>`_
    """

    X_train = X.iloc[rolling["tr"][0]: rolling["tr"][1]]
    y_train = y.iloc[rolling["tr"][0]: rolling["tr"][1]]
    X_test = X.iloc[rolling["ts"][0]: rolling["ts"][1]]
    y_test = y.iloc[rolling["ts"][0]: rolling["ts"][1]]

    return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}


def fit_model_hyperopt(
    params: Dict,
    estimator: str,
    kfolds: int,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> Dict:
    """
    Function to perform hyperparameter tunning using training data.


    :param params: The selected hyperameter values from the search space.
    :type params: Dict

    :param estimator: The name of classification estimator.
    :type estimator: str

    :param kfolds: The number of folds of cross-validation.
    :type kfolds: int

    :param X_train: The training Dataframe.
    :type X_train: pd.DataFrame

    :param y_train: The labels from the training set.
    :type y_train: pd.DataFrame


    :return: Loss (value to be minimized) and status.
    :rtype: Dict


    .. seealso:: Tested Machine Learning Classification models

        1. `Random Forest \
            <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
        2. `Decision Tree \
            <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier>`_
        3. `Gradient Boosted Tree \
            <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
        4. `Logistic Regression \
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
        5. `Support Vector Machine \
            <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
        6. `'Multilayer Perceptron \
            <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_
        7. `XGBoost \
            <https://xgboost.readthedocs.io/en/stable/python/python_api.html>`_
    """

    if estimator == "RF":
        # Random Forest Classifier
        print(
            "Random Forest -> max_depth: {0} | min_samples_split: {1}".format(
                int(params['max_depth']), int(params['min_samples_split'])))

        model = RandomForestClassifier(
            n_estimators=200, max_depth=int(params['max_depth']), min_samples_split=int(params['min_samples_split']))

    elif estimator == "DT":
        # Decision Tree Classifier
        print(
            "Decision Tree Classifier -> max_depth: {0} | min_samples_split: {1}".format(
                int(params['max_depth']), int(params['min_samples_split'])))

        model = DecisionTreeClassifier(max_depth=int(
            params['max_depth']), min_samples_split=int(params['min_samples_split']))

    elif estimator == "LR":
        # Logistic Regression
        print("Logistic Regression -> regParam: {0} | l1_ratio: {1}".format(
            float(params['C']), float(params['l1_ratio'])))

        model = LogisticRegression(max_iter=100, C=float(
            params['C']), penalty='elasticnet', l1_ratio=float(params['l1_ratio']), solver='saga')

    elif estimator == "GBT":
        # Gradient-Boosted Tree Classifier
        print("Gradient-Boosted Tree Classifier -> max_depth: {0} | min_samples_split: {1}".format(
            int(params['max_depth']),
            int(params['min_samples_split']))
        )

        model = GradientBoostingClassifier(n_estimators=200, max_depth=int(
            params['max_depth']), min_samples_split=int(
            params['min_samples_split']))
    elif estimator == "SVC":
        # Linear Support Vector Machine
        print("Support Vector Machine -> C: {0} | kernel: {1}".format(
            float(params['C']), params['kernel']))

        model = SVC(C=float(params['C']), kernel=params['kernel'])

    elif estimator == "XGBT":
        # XGBoost
        print("XGBT -> max_depth: {0} | max_bin: {1} | eta: {2}".format(
            float(int(params['max_depth'])), int(params['max_bin']), float(params['eta'])))

        model = XGBClassifier(
            max_depth=int(params['max_depth']), max_bin=int(params['max_bin']), eta=float(params['eta']))

    elif estimator == "MLP":
        # Multilayer Perceptron Classifier
        print(
            "Multilayer Perceptron -> alpha: {0} | learning_rate: {1}".format(float(params['alpha']),
                                                                              params['learning_rate']))

        inputLayer = np.size(X_train, 1)
        hiddenLayer = int(round(inputLayer / 2))

        # print("Input Layer: {}".format(inputLayer))

        hidden_layers = (hiddenLayer,)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=500,
            alpha=float(params['alpha']),
            solver='sgd',
            learning_rate=params['learning_rate']
        )

    cval = cross_val_score(model, X_train, y_train,
                           scoring='roc_auc', cv=kfolds)

    auc = cval.mean()
    # Because fmin() tries to minimize the objective, this function must return the negative auc.
    return {'loss': -auc, 'status': STATUS_OK}


def fit_model_h(
    params: Dict,
    estimator: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Dict:
    """
    Function to train models using training data and predict using testing data.


    :param params: The selected hyperameter values from the search space.
    :type params: Dict

    :param estimator: The name of classification estimator.
    :type estimator: str

    :param X_train: The training DataFrame.
    :type X_train: pd.DataFrame

    :param y_train: The labels from the training set.
    :type y_train: pd.DataFrame

    :param X_test: The test DataFrame.
    :type X_test: pd.DataFrame


    :return: The predicted values, prediction probabilities, model fitted, elapsed time to train and predict.
    :rtype: Dict


    .. seealso:: `Tested Machine Learning Classification models  \
        <https://scikit-learn.org/stable/tutorial/basic/tutorial.html>`_

        1. `Random Forest \
            <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
        2. `Decision Tree \
            <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier>`_
        3. `Gradient Boosted Tree \
            <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
        4. `Logistic Regression \
            <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
        5. `Support Vector Machine \
            <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
        6. `'Multilayer Perceptron \
            <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_
        7. `XGBoost \
            <https://xgboost.readthedocs.io/en/stable/python/python_api.html>`_
    """

    if estimator == "RF" or estimator == "DT" or estimator == "GBT":
        search_space = {
            'max_depth': [2, 5, 10, 20, 30],
            'min_samples_split': [2, 6, 10]
        }

        best_maxDepth = int(params['max_depth'])
        best_minSplit = int(params['min_samples_split'])

        Model_maxDepth = search_space['max_depth'][best_maxDepth]
        Model_minSplit = search_space['min_samples_split'][best_minSplit]

        if estimator == "RF":
            # Random Forest Classifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=Model_maxDepth, min_samples_split=Model_minSplit)

        elif estimator == "DT":
            # Decision Tree Classifier
            model = DecisionTreeClassifier(
                max_depth=Model_maxDepth, min_samples_split=Model_minSplit)

        elif estimator == "GBT":
            model = GradientBoostingClassifier(
                n_estimators=200, max_depth=Model_maxDepth, min_samples_split=Model_minSplit)

    elif estimator == "LR" or estimator == "SVC":
        search_space1 = {
            'C': [0.01, 0.1, 0.5, 1.0, 2.0],
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        }

        best_C = int(params['C'])
        Model_C = cast(List, search_space1['C'])[best_C]

        if estimator == "LR":
            best_l1_ratio = int(params['l1_ratio'])
            Model_l1_ratio = cast(List, search_space1['l1_ratio'])[
                best_l1_ratio]

            # Logistic Regression
            model = LogisticRegression(penalty='elasticnet',
                                       max_iter=200, C=Model_C, l1_ratio=Model_l1_ratio, solver='saga')

        elif estimator == "SVC":
            best_kernel = params['kernel']
            Model_kernel = cast(List, search_space1['kernel'])[best_kernel]

            # Support Vector Machine
            model = SVC(C=Model_C, probability=True, kernel=Model_kernel)

    elif estimator == 'XGBT':
        search_space5 = {
            'eta': [0.0, 0.25, 0.5, 0.75, 1.0],
            'max_depth': [2, 5, 10, 20, 30],
            'max_bin': [10, 20, 40, 80, 100]
        }

        best_maxDepth = int(params['max_depth'])
        best_maxBin = int(params['max_bin'])
        best_eta = int(params['eta'])

        Model_eta = cast(List, search_space5['eta'])[best_eta]
        Model_maxDepth = cast(List, search_space5['max_depth'])[best_maxDepth]
        Model_maxBin = cast(List, search_space5['max_bin'])[best_maxBin]

        model = XGBClassifier(max_depth=Model_maxDepth, max_bin=Model_maxBin,
                              eta=Model_eta)

    elif estimator == "MLP":
        search_space6 = {
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }

        best_alpha = int(params['alpha'])
        best_learningRate = params['learning_rate']

        Model_alpha = cast(List, search_space6['alpha'])[best_alpha]
        Model_learningRate = cast(List, search_space6['learning_rate'])[
            best_learningRate]

        # Multilayer Perceptron Classifier
        inputLayer = np.size(X_train, 1)
        hiddenLayer = round(inputLayer / 2)

        print("Input Layer: {}".format(inputLayer))

        hidden_layers = (hiddenLayer,)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=500,
            alpha=Model_alpha,
            solver='sgd',
            learning_rate=Model_learningRate
        )

    start_time = time.time()
    fitmodel = model.fit(X_train, y_train)
    end_time = time.time()
    time_elapsed1 = (end_time - start_time)

    start_time = time.time()
    results = fitmodel.predict(X_test)
    end_time = time.time()
    time_elapsed2 = (end_time - start_time)

    probs = fitmodel.predict_proba(X_test)

    return {
        'y_pred': results,
        'probs': probs,
        'model': fitmodel,
        'time_train': time_elapsed1,
        'time_predict': time_elapsed2
    }


def model_evaluation_metrics(
    y_test: pd.DataFrame,
    y_pred: pd.DataFrame
) -> Dict:
    """
    Function to calculate the several classification evaluation metrics.


    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param y_pred: The predicted labels.
    :type y_pred: pd.DataFrame


    :return: The models metrics: AUC, Accuracy, F1-Score, Recall and Precision.
    :rtype: Dict

    .. seealso:: `Metrics and scoring: quantifying the quality of predictions  \
        <https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics>`_

        1. `AUC Score \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`_
        2. `Accuracy Score \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`_
        3. `F1 Score \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_
        4. `Precision Score \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_
        5. `Recall Score \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`_
    """

    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return {
        'AUC': auc,
        'Accuracy': acc,
        'F1-Score': f1,
        'Recall': recall,
        'Precision': precision
    }


def get_roc_curve_metrics(
    y_test: pd.DataFrame,
    probability: Any
) -> Dict:
    """
    Function to compute the Receiver operating characteristic (ROC) curve.


    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param probability: The predicted labels probabilities.
    :type probability: Any


    :return: The fpr, tpr metrics, thresholds and auc_roc.
    :rtype: Dict


    .. seealso:: `Receiver operating characteristic (ROC) metrics  \
        <https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html>`_

        1. `ROC Curve \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve>`_
        2. `ROC AUC Score \
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_
    """
    probability_class_1 = probability[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, probability_class_1)
    roc_auc = auc(fpr, tpr)

    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'roc_auc': roc_auc}
