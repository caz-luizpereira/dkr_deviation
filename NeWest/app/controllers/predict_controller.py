"""
predict_controller.py module
"""

import pickle

def predict_model(body):

    """
    Function that predict if the process will be blocked

    :param body: json object with the data that will be used for the prediction
    :type body: json

    :return: Prediction Classification [0] or [1]
    :rtype: str
    """

    # load the standarization model
    filename_standarization = "./Models/Model_Training/Standarization_Model/Scaler_Model.sav"

    # load the model
    filename_model = "./Models/Model_Training/Trained_Model/XGBOOST_Predict_Model.sav"

    loaded_standarization = pickle.load(open(filename_standarization, 'rb'))

    body = loaded_standarization.transform(body["features"])

    loaded_model = pickle.load(open(filename_model, 'rb'))

    # prediction
    result = loaded_model.predict(body)

    result = str(result)

    return result