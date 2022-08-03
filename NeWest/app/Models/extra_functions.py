import pandas as pd
from typing import List, Optional, Any


def save_dataframe_to_csv(
    df: pd.DataFrame,
    model: str,
    file_name: str,
    location: Optional[str] = None
) -> None:
    """
    Function save Dataframe to csv file


    :param df: The Dataframe of Weighings.
    :type df: pd.DataFrame

    :param model: The name of tested model.
    :type model: str

    :param location: The location to save the feature importance plot. By default is None
    :type location: Optional[str]


    :return: None.
    :rtype: None


    .. seealso:: `Write object to a comma-separated values (csv) file  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`_
    """

    if location is None:
        location = 'static/results'

    df.to_csv('{0}/{1}/{1}_{2}.csv'.format(location,
              model, file_name), sep=';', index=False)


def number_input_validation(
    message: str
) -> int:
    """
    Function to validate input


    :param message: The message to be displayed.
    :type message: str


    :return: Number introduced by user.
    :rtype: int
    """

    while True:
        try:
            number = int(input(message))
        except ValueError:
            print("Select a valid option! Try again.")
            continue
        else:
            return number
            break


def show_menu(
    message: str,
    options: List
) -> int:
    """
    Function to display the menu


    :param message: The message to be displayed.
    :type message: str


    :return: Number introduced by user.
    :rtype: int
    """

    print("\n########################################")
    print("  NEWEST - {} ".format(message))
    print("########################################\n")
    for i in range(0, len(options), 1):
        print("  {} - {}".format(i, options[i]))
    print("\n########################################\n")
    selected = number_input_validation("Select a option: ")
    while selected < 0 or selected >= len(options):
        print(
            "Select a valid option [0-{}]! Try again.".format(len(options) - 1))
        selected = number_input_validation("Select a option: ")
    return selected


def save_predictions(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    y_pred: List,
    probs: Any,
    model: str,
    iteration: int,
    location: Optional[str] = None
) -> None:
    """
    Function to save prediction in csv file


    :param X_test: The test DataFrame.
    :type X_test: pd.DataFrame

    :param y_test: The labels from the test set.
    :type y_test: pd.DataFrame

    :param y_pred: The prediced labels.
    :type y_pred: pd.DataFrame

    :param probs: The prediction probabilities of each class.
    :type probs: Any

    :param model: The name of tested model.
    :type model: str

    :param iteration: The number of RW iteration.
    :type iteration: int

    :param location: The location to save prediction in csv file format. By default is None
    :type location: Optional[str] = None


    :return: None.
    :rtype: None


    .. seealso:: `Merge DataFrame  \
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html>`_
    """

    if location is None:
        location = 'static/results'

    predicted_df = pd.DataFrame(data=y_pred, columns=['y_pred'],
                                index=X_test.index.copy())

    probability_class_0 = probs[:, 0]
    probability_class_1 = probs[:, 1]

    probs_df0 = pd.DataFrame(data=probability_class_0, columns=['Prob0'],
                             index=X_test.index.copy())

    probs_df1 = pd.DataFrame(data=probability_class_1, columns=['Prob1'],
                             index=X_test.index.copy())

    df_out = pd.merge(X_test, y_test, how='left', left_index=True,
                      right_index=True)

    df_out = pd.merge(df_out, predicted_df, how='left', left_index=True,
                      right_index=True)

    df_out = pd.merge(df_out, probs_df0, how='left', left_index=True,
                      right_index=True)

    df_out = pd.merge(df_out, probs_df1, how='left', left_index=True,
                      right_index=True)

    save_dataframe_to_csv(
        df_out, model, '{}'.format(iteration))
