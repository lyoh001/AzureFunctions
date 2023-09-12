import datetime
import logging
import warnings
from pickle import load

import azure.functions as func
import numpy as np
import pandas as pd
import scipy.stats as stat
import tflite_runtime.interpreter as tflite

warnings.filterwarnings("ignore")

preprocessor = load(open("mltelecom/preprocessor/preprocessor.pkl", "rb"))
model = tflite.Interpreter(model_path="mltelecom/model/model.tflite")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")
    payload = {
        k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()
    }
    city_name = next(iter(payload["city_name"]))
    time_period = (
        datetime.datetime.strptime(payload["time_period"][0], "%Y-%m")
        - datetime.datetime.strptime("2022-09", "%Y-%m")
    ).days // 29
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
    model.allocate_tensors()
    input_index = model.get_input_details()[0]["index"]
    input_tensor = preprocessor.transform(X).astype("float32")
    output_details = model.get_output_details()
    model.set_tensor(input_index, input_tensor)
    model.invoke()
    prediction = model.get_tensor(output_details[0]["index"])[0][0]
    output = f"Based on the input attributes of city_name (ie '{city_name}'), data_size (ie '{data_size} Gb'), and valid_period (ie '{int(valid_period)} days'), the AI model predicts the monthly plan revenue to be ₹{prediction:,.2f} (Crores) post the 5G rollout. The model is trained on a dataset of historical data, and it uses this data to learn the relationships between the input attributes and the output. The accuracy of the model will depend on the quality of the training data. If the training data is representative of the real world, then the model will be more accurate. However, if the training data is not representative of the real world, then the model may be less accurate. The model can be used to help telecommunication companies make better decisions about their pricing and marketing strategies. For example, the model can be used to identify cities where there is a high demand for 5G plans, or to identify time periods when there is a seasonal increase in demand for mobile data."
    logging.info(f"output: {output}")

    return func.HttpResponse(status_code=200, body=output)
