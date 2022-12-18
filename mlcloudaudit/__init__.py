import calendar
import datetime
import io
import logging
import warnings
from pickle import load

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import azure.functions as func

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

X_preprocessor = load(open("mlcloudaudit/preprocessor/X_preprocessor.pkl", "rb"))
y_preprocessor = load(open("mlcloudaudit/preprocessor/y_preprocessor.pkl", "rb"))
model = keras.models.load_model("mlcloudaudit/model")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")
    payload = {
        k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()
    }
    try:
        time_period = (datetime.datetime.strptime(payload["time_period"][0], "%Y-%m")- datetime.datetime.strptime("2022-09", "%Y-%m")).days // 29
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input.",
        )
    N_PAST = 5
    N_FUTURE = 1
    ASFREQ = "MS"
    FREQ = "1m"
    DATE_COL = "Date"
    y_label = "Bill"

    df = pd.read_csv(
        io.StringIO(
            """Date,Bill
            2022-06,334230
            2022-07,344321
            2022-08,351373
            2022-09,346814
            2022-10,387434
            2022-11,394204
            2022-12,401308"""
        )
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m")
    df = df.set_index(DATE_COL).sort_index().asfreq(ASFREQ)
    X_test_scaled = X_preprocessor.transform(df)
    X_test = np.array(
        [
            X_test_scaled[i : i + N_PAST]
            for i in range(len(X_test_scaled) - N_PAST - N_FUTURE + 1)
        ]
    )
    pred_future, current_batch = [], X_test[-1:]
    for _ in range(N_PAST):
        current_pred = model.predict(current_batch)[0]
        pred_future.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    df_future_pred = pd.DataFrame(
        y_preprocessor.inverse_transform(pred_future),
        columns=[y_label],
        index=pd.date_range(df.index[-1], periods=N_PAST, freq=FREQ, name=DATE_COL),
    )
    logging.info(df_future_pred)
    
    return func.HttpResponse(
        status_code=200,
        body=f"{df_future_pred}"
    )