import pandas as pd
import pandas_profiling
from typing import List, Union, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def drop_columns(
    df: pd.DataFrame,
    columns: List
) -> pd.DataFrame:
    """
    Function drop columns from Dataframe


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The list of column to drop from Dataframe.
    :type columns: List


    :return:  Dataframe with columns dropped.
    :rtype: pd.DataFrame


    .. seealso:: `DataFrame: Drop specified labels from rows or columns  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html>`_
    """

    df = df.drop(columns, axis=1)

    return df


def replace_column_value(
    df: pd.DataFrame,
    column: str,
    oldValue: Union[str, int],
    newValue: Union[str, int],
) -> pd.DataFrame:
    """
    Function to replace column value from Dataframe


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The column name to replace values.
    :type columns: str


    :return: Dataframe with columns values replaced.
    :rtype: pd.DataFrame


    .. seealso:: `DataFrame: Replace values given in to_replace with value  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html>`_
    """

    df[column] = df[column].str.replace(oldValue, newValue)

    return df


def normalization(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple:
    """
    Function to normalize the dataset using the min max scaler [0, 1]


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The list of column to normalize from Dataframe.
    :type columns: List


    :return:  Tuple with scaler model, train and test data normalized.
    :rtype: Tuple


    .. seealso:: `Scale input vectors individually to unit norm (vector length)  \
        <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html>`_
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return scaler, train, test


def standardization(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple:
    """
    Function to stardardize the dataset using the z-score scaler (StandardScaler)


    :param df: The training data.
    :type df: pd.DataFrame


    :return: Tuple with scaler model, train and test data stardardized.
    :rtype: Tuple


    .. seealso:: `Standardize features by removing the mean and scaling to unit variance  \
        <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_
    """

    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler = scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return scaler, train, test


def one_hot_encoding_categorical(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Function to encoding the selected categorical features from the dataset


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The list of column to encode from Dataframe.
    :type columns: List


    :return: Dataframe with categorical features encoded.
    :rtype: pd.DataFrame


    .. seealso:: `Encode categorical features as a one-hot numeric array  \
        <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html>`_
    """

    cat_cols_encoded = []

    for col in columns:
        cat_cols_encoded += [
            f"{col[0]}_{cat}" for cat in list(df[col].unique())]

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    encoded_cols = encoder.fit_transform(df[columns])
    df_enc = pd.DataFrame(encoded_cols, columns=cat_cols_encoded)
    df_oh = df.join(df_enc)

    return df_oh


def rolling_window_operation(
    df: pd.DataFrame,
    columns: List[str],
    window: int,
    lag: int
) -> pd.DataFrame:
    """
    Function calculate the average of deviation using rolling window from the dataset


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: Rolling Windows Columns
    :type columns: List

    :param columns[0]: The column related to values to calculate the average.
    :type columns[0]: str

    :param columns[1]: The new column that will be created to store the average values.
    :type columns[1]: str

    :param columns[2]: The new column that will be created to store the average values.
    :type columns[2]: str

    :param window: The size of the window to calculate the average.
    :type window: int

    :param lag: The lag of rolling window (period to be skipped when calculating the average)
    :type lag: int


    :return: Dataframe with average column calculate over the rolling window.
    :rtype: pd.DataFrame


    .. seealso:: `Rolling window calculations  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html>`_

        1. `DataFrame: Group DataFrame \
            <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html>`_
        2. `DataFrame: Shift index by desired number of periods \
            <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html>`_
        3. `DataFrame: Mean of the values over the requested axis \
            <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html>`_
    """

    df[columns[1]] = (df.groupby(columns[2])[columns[0]].transform(
        lambda x: x.rolling(window).mean().shift(lag)))

    return df


def weekly_daily_average(
    df: pd.DataFrame,
    columns: List[str],
    period: str,
) -> pd.DataFrame:
    """
    Function to calculate the average of deviation per type of vehicle over time


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: Rolling Windows Columns
    :type columns: List

    :param columns[0]: The column related to values to calculate the average.
    :type columns[0]: str

    :param columns[1]: The new column that will be created to store the average values.
    :type columns[1]: str

    :param columns[2]: The first column that will used to groupby.
    :type columns[2]: str

    :param columns[3]: he second column that will used to groupby.
    :type columns[3]: str

    :param period:  The period (daily or weekly to calculate the average of deviation)
    :type period: str


    :return: Dataframe with weekly average column calculate over time (Tare date).
    :rtype: pd.DataFrame


    .. seealso:: `Rolling window calculations  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html>`_

        1. `DataFrame: Group DataFrame \
            <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html>`_
        2. `Series: DatetimeArray/Index to PeriodArray/Index \
            <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.to_period.html>`_
    """

    df[columns[1]] = (df.groupby([df[columns[2]].dt.to_period(
        period), df[columns[3]]])[columns[0]].transform('mean'))

    return df


def dataframe_fillna(
    df: pd.DataFrame,
    column: str,
    value: Union[str, int]
) -> pd.DataFrame:
    """
    Function to replace NaN to a defined value that could be string ou integer


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The column to fillna.
    :type columns: str


    :return: Dataframe with NaN replaced.
    :rtype: pd.DataFrame.


    .. seealso:: `DataFrame: Fill NA/NaN values using the specified method  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html>`_
    """

    df[column] = df[column].fillna(value)

    return df


def dataframe_replace(
    df: pd.DataFrame,
    column: str,
    value: Union[str, int],
    newvalue: Union[str, int]
) -> pd.DataFrame:
    """
    Function to replace a given value to a defined new value (these values could be string ou integer)


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The column to replace.
    :type columns: str


    :return: Dataframe with given value replaced.
    :rtype: pd.DataFrame

    .. seealso:: `DataFrame: Replace values given in to_replace with value  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html>`_
    """

    df[column] = df[column].str.replace(value, newvalue)

    return df


def date_transformation(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Function for date transformation in datetime format


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The column to be transformed.
    :type columns: str


    :return: Dataframe with given value replaced.
    :rtype: pd.DataFrame


    .. seealso:: `Convert argument to datetime  \
        <https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html>`_
    """

    df[column] = pd.to_datetime(df[column], format='%Y/%m/%d %H:%M:%S.%f')

    return df


def get_datetime_attributes(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Function for getting datetime format attributes


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param columns: The column to get attributes.
    :type columns: str


    :return: Dataframe new columns created from a given datatime column (Hour, Day, Day of week, Month).
    :rtype: pd.DataFrame

    .. seealso:: `Convert argument to datetime  \
        <https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html>`_

        1. `Hour from Datetime \
            <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.hour.html>`_
        2. `Day from Datetime \
            <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.day.html>`_
        3. `Day of Week from Datetime \
            <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.dayofweek.html>`_
        4. `Month from Datetime \
            <https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.month.html>`_
    """

    df['Hour'] = df[column].dt.hour
    df['Day'] = df[column].dt.day
    df['DayOfWeek'] = df[column].dt.day_of_week
    df['Month'] = df[column].dt.month

    return df


def generate_data_profiling_report(
    df: pd.DataFrame,
    location: Optional[str] = None,
) -> None:
    """
    Function for generate data profiling report from the given dataset.


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param location: The location to the data profiling report. By default is None
    :type location: Optional[str]


    :return: None.
    :rtype: None


    .. seealso:: `Pandas Profiling  \
        <https://pandas-profiling.ydata.ai/docs/master/index.html>`_
    """

    if location is None:
        location = 'static/report'

    report = pandas_profiling.ProfileReport(df)
    report.to_file('{}/data_profiling.html'.format(location))
