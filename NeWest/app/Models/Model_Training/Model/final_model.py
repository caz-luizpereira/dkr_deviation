import statistics
import pandas as pd
from typing import Dict
from functools import partial
# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import data_ingestion as d_ingest
import data_preparation as d_prep
import data_visualization as d_visual
import modeling_evaluation as mod_eval
import extra_functions as extra_func
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor

def data_preparation(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to perform data preparation tasks


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame


    :return: Dataframe transformed.
    :rtype: pd.DataFrame
    """

    # Columns to be dropped (unrequired columns)
    columns_drop = [
        'TipoDoc', 'estado', 'bruto', 'Liquido', 'DataCriacao',
        'Dataentrada', 'DataInicioOperacao', 'DataFimOperacao',
        'BrutoData', 'DataFecho', 'CodEntidade', 'CodMotorista', 'NomeMotorista'
    ]

    # Columns to be renamed
    columns_rename = {
        'PostoOperacao': 'Station',
        'TipoViatura': 'Vehicle_Type',
        'DescProduto': 'Product',
        'CodProduto': 'CodProduct',
        'Tara': 'Tare',
        'Matricula': 'Plate',
        'qtdpedida': 'Qty_Ordered',
        'TaraData': 'Tare_Date',
        'percDiff': 'Deviation'
    }

    # Remove special character from column names
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '', regex=True)

    # Filter all rows in which the net weighing is less than or equal to zero
    df = df.drop(df[df['Liquido'] <= 0].index)

    # Remove/Dropp unrequired columns
    df = d_prep.drop_columns(df, columns_drop)

    # Rename the columns
    df.rename(columns=columns_rename, inplace=True)

    # Replace ',' for '.' in order to afterwards convert Deviation to float
    df = d_prep.replace_column_value(df, 'Deviation', ',', '.')

    # Transform TaraData to Datetime format type
    df = d_prep.date_transformation(df, 'Tare_Date')

    # Fillna in the PostoOperacao attribute by "Unknown"
    df = d_prep.dataframe_fillna(df, 'Station', 'Unknown')

    # Convert Qty_Ordered and Deviation attributes to float format type
    df[['Qty_Ordered', 'Deviation']] = df[[
        'Qty_Ordered', 'Deviation']].astype(float)

    # categorical_columns = ['Station']

    # df = d_prep.one_hot_encoding_categorical(df, categorical_columns)

    return df


def feature_engineering(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to perform data preparation tasks


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame


    :return: The processed DataFrame.
    :rtype: pd.DataFrame
    """

    # Create new Dataframe columns from the datatime attributes (e.g., Hour, Day, Month)
    df = d_prep.get_datetime_attributes(df, 'Tare_Date')

    # Defining Supervised and Unsupervised Work Shifts
    df["Inspection"] = df['Hour'].apply(
        lambda x: 0 if x >= 6 and x < 18 else 1)

    # Create Labels for classification
    df["Block"] = df['Deviation'].apply(
        lambda x: 0 if x >= -2 and x <= 2 else 1)

    # Sort Dataframe by Tare date
    df = df.sort_values(by="Tare_Date")

    # Create new features from existing dataset variables
    # df = d_prep.rolling_window_mean(df, ['Deviation', 'Average_Deviation_Vehicle_W5', 'Vehicle_Type'], 5, 1)

    df = d_prep.rolling_window_operation(
        df, ['Deviation', 'Average_Deviation_Station', 'Station'], 5, 1)

    # df = d_prep.weekly_daily_average(df, ['Deviation', 'Average_Vehicle_Weekly', 'Tare_Date', 'Vehicle_Type'], 'W')

    df = d_prep.weekly_daily_average(
        df, ['Deviation', 'Average_Station_Weekly', 'Tare_Date', 'Station'], 'W')

    # df = d_prep.weekly_daily_average(df, ['Deviation', 'Average_Vehicle_Hourly', 'Tare_Date', 'Vehicle_Type'], 'H')

    df = d_prep.weekly_daily_average(
        df, ['Deviation', 'Average_Station_Hourly', 'Tare_Date', 'Station'], 'H')

    # Drop all NaN from dataset
    df = df.dropna()
    df = df.reset_index(drop=True)

    df = d_prep.rolling_window_operation(
        df, ['Block', 'Percentage_Blocks', 'Plate'], 5, 1)

    df['Percentage_Blocks'] = df['Percentage_Blocks'].fillna(0)

    return df


def vif_calculation(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to calculate the Multicollinearity between the selected features
    using Variable Inflation Factors (VIF), in order to determines the strength
    of the correlation of a variable     with a group of other independent variables
    in a dataset.
    OBS: VIF starts usually at 1 and anywhere exceeding 10 indicates
    high multicollinearity between the independent variables.


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame


    :return: VIF Dataframe with the name of each features and the corresponding VIF.
    :rtype: pd.DataFrame

    .. seealso:: `Variance inflation factor (VIF)  \
        <https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html>`_
    """

    vif = pd.DataFrame()
    vif["Features"] = df.columns
    vif["VIF"] = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]

    return vif


def modeling(
    df: pd.DataFrame
) -> Dict:
    """
    Function to perform modeling and evaluation tasks


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame


    :return:  The models metrics: AUC, Time to train, Time to predict, Accuracy,
        F1 score, Recall and Precision for each tested models regarding each Rolling Window (RW) iterations.
    :rtype: Dict

    .. seealso:: `Hyperopt: Distributed Asynchronous Hyper-parameter Optimization \
        <http://hyperopt.github.io/hyperopt/>`_

        1. `Defining a Search Space \
            <http://hyperopt.github.io/hyperopt/getting-started/search_spaces/>`_
        2. `SMOTE \
            <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html>`_
    """

    # Drop unusead DataFrame variables
    df = df.drop([
        "Vehicle_Type",
        "CodProduct",
        "Product",
        "Deviation",
        "Station",
        "Tare_Date",
        "Plate",
        "Tare"
    ], axis=1)

    # Definition of 5-Fold Cross-validation and Bayesian Optimization over 10 iterations
    kfolds = 5
    n_iter = 20

    # Search space of parameters (possible values of parameters)
    search_space = {
        'eta': hp.choice('eta', [0.0, 0.25, 0.5, 0.75, 1.0]),
        'max_depth': hp.choice('max_depth', [2, 5, 10, 20, 30]),
        'max_bin': hp.choice('max_bin', [10, 20, 40, 80, 100])
    }

    models = {
        "XGBT": ['XGBoost', search_space, True]
    }

    dict_auc = {}
    dict_time_train = {}
    dict_time_predict = {}
    dict_accuracy = {}
    dict_f1_score = {}
    dict_recall = {}
    dict_precision = {}


    # SMOTE - Synthetic Minority Over-sampling Technique
    sm = SMOTE()
    df_size = len(df)

    # Separate the label from the original dataFrame
    X = df.drop("Block", axis=1)
    y = df["Block"]

    #Train and Test Split
    X_train ,X_test, y_train, y_test = train_test_split(X, y, shuffle = False, stratify = None)

    # summarize class distribution
    counter = Counter(y_train)
    print("Before SMOTE -> " + str(counter))

    #Smote Aplication
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # summarize class distribution
    counter = Counter(y_train)
    print("After SMOTE -> " + str(counter))

    #Data Standarization
    scaler = d_prep.standardization(X_train, X_test)
    scaler_model = scaler[0]
    X_train_scaled = scaler[1]
    X_test_scaled = scaler[2]
    

    df_fixe = pd.DataFrame(X_test_scaled)
    X.head(4).to_csv('out.csv')  
 
    model = XGBClassifier(max_depth=4, max_bin=3, eta=1)

    print("\nFit {}...".format("XGBT"))

    best_Parameters = {'eta': 1, 'max_bin': 1, 'max_depth': 3}

    results = mod_eval.fit_model_h(params=best_Parameters,estimator="XGBT", X_train=X_train_scaled, y_train=y_train,X_test=X_test_scaled)

    predictions = results['y_pred']
    model = results['model']

    filename = '../Standarization_Model/Scaler_Model.sav'
    pickle.dump(scaler_model, open(filename, 'wb'))

    # save the model to disk
    filename = '../Trained_Model/XGBOOST_Predict_Model.sav'
    pickle.dump(model, open(filename, 'wb'))

    probability = results['probs']
    time_elapsed1 = results['time_train']
    time_elapsed2 = results['time_predict']
    
    model_metrics = mod_eval.model_evaluation_metrics(y_test, predictions)

    areaUnderROC = model_metrics['AUC']
    accuracy = model_metrics['Accuracy']
    f1_score = model_metrics['F1-Score']
    recall = model_metrics['Recall']
    precision = model_metrics['Precision']

    print("\n################### XGBOOST - Results #####################")
    print("Time-Elapsed Fit - '{}'".format(time_elapsed1))
    print("Time-Elapsed Transform - '{}'".format(time_elapsed2))
    print("Area under ROC - '{}'".format(areaUnderROC))
    print("Accuracy - '{}'".format(accuracy))
    print("F1-Score - '{}'".format(f1_score))
    print("Recall - '{}'".format(recall))
    print("Precision - '{}'".format(precision))

    return results

def newest_main() -> None:
    """
    Main Fuction


    :return: None.
    :rtype: None
    """

    args = "../../static/files/dataset.csv"
    print(args)
    pd.set_option('display.max_rows', None)
    # ags[0] -> Machine deviation file;
    if d_ingest.file_exists(args) is True:

        # Read machine deviation data
        df = d_ingest.read_file_data(args)

        df = data_preparation(df)

        df = feature_engineering(df)

        ml_results = modeling(df)

        listaIdx = list(range(1, 21))

        models = {
            "XGBT": 'XGBoost'
        }


    print("\n----------------- Finish -----------------")

# python app.py -f static/files/dataset.csv
if __name__ == "__main__":
    newest_main()

