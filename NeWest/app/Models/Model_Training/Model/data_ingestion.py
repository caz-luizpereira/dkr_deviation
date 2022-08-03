import argparse
from os import path  # https://docs.python.org/3/library/os.path.html
import pandas as pd
from typing import List


def parse_command_line(
) -> List:
    """
    Function to parser the command-line options


    :return: The arguments passed through the command-line (i.e., the file location).
    :rtype: List

    .. seealso:: `Parser for command-line options, arguments and sub-commands  \
        <https://docs.python.org/3/library/argparse.html>`_
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-file_path",
        metavar='-f',
        type=str,
        nargs="+",
        help="directory from where read data regarding weighings and machine deviation",
    )
    parser_args = parser.parse_args()

    return parser_args.file_path


def file_exists(
    location: str
) -> bool:
    """
    Function to verify if file exists


    :param location: The file location.
    :type location: str


    :return: True is file exists and otherwise False.
    :rtype: bool


    .. seealso:: `Common pathname manipulations  \
        <https://docs.python.org/3/library/pathlib.html#methods>`_
    """

    # Verify if is not a file according to the given file path
    if not path.isfile(location):
        return False

    return True


def read_file_data(
    location: str
) -> pd.DataFrame:
    """
    Function of read a all file lines to a list. Each index of a list corresponds to one line


    :param location: The file location.
    :type location: str


    :return: Pandas Dataframe of file contents read.
    :rtype: pd.DataFrame

    .. seealso:: `Read a comma-separated values (csv) file into DataFrame  \
        <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html>`_
    """

    file_data = pd.DataFrame
    try:
        file_data = pd.read_csv(r'{}'.format(location),
                                on_bad_lines='skip', sep=';', encoding='latin-1')
    except IOError:
        print("An Exception occurred when reading the file!!")

    return file_data
