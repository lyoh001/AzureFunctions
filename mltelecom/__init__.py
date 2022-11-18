import datetime
import logging
import warnings
from pickle import load

import azure.functions as func
import numpy as np
import pandas as pd
import scipy.stats as stat
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

preprocessor = load(open("mltelecom/preprocessor/preprocessor.pkl", "rb"))
model = keras.models.load_model("mltelecom/model")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")
    payload = {
        k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()
    }
    city_name = next(iter(payload["city_name"]))
    time_period = (datetime.datetime.strptime(payload["time_period"][0], "%Y-%m")- datetime.datetime.strptime("2022-09", "%Y-%m")).days // 29
    try:
        data_size = float(next(iter(payload["data_size"])))
        if not 0.1 <= data_size <= 30:
            return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 0.1 and 30.",
        )
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 0.1 and 30.",
        )
    try:
        valid_period = float(next(iter(payload["valid_period"])))
        if not 1 <= valid_period <= 180:
            return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 1 and 180.",
        )
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 1 and 180.",
        )
    X = pd.DataFrame(
        {
            "city_name": [city_name],
            "time_period": [time_period],
            "data_size": [data_size],
            "valid_period": [valid_period],
            "before/after_5g": ["After 5G"],
        }
    )
    X["data_size"] = stat.boxcox(X["data_size"].iloc[:1], -0.15236159025676418)[0]
    X["valid_period"] = stat.boxcox(X["valid_period"].iloc[:1], 0.2613605015797948)[0]
    prediction = model.predict(preprocessor.transform(X))[0][0]
    logging.info(X)
    logging.info(f"prediction: {prediction}")
    
    return func.HttpResponse(
        status_code=200,
        body=f"{prediction:.2f} Crores Rs"
    )