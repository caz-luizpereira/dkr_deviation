"""App.py module"""

from flask import Flask, request
from flask_cors import CORS
from threading import Timer, Thread
import pickle
import controllers.predict_controller as predict_controller

app = Flask(__name__)
CORS(app)

##### ROUTES #####
@app.route('/', methods=['GET'])
def routeRoot() -> str:
    """
    Function Main root

    :return:  MicroService Message
    :rtype: str
    """
    return "MicroService - Bag Loading Machine Derivations"


@app.route('/predict', methods=['POST'])
def derivation_predict() -> str:
    
    """
    Function that predict if the process will be blocked

    :param body: json object with the data that will be used for the prediction
    :type body: json

    :return: Prediction Classification [0] or [1]
    :rtype: str
    """
    body = request.get_json()

    prediction = predict_controller.predict_model(body)

    prediction = str(prediction[1])
    
    return prediction

# Run the application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10003)
