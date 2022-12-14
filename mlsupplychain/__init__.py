import datetime
import logging
import warnings
from pickle import load

import azure.functions as func
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings("ignore")

df = pd.read_csv(
    "mlsupplychain/data/fact_order_lines_merged.csv", delimiter=",", thousands=","
)
preprocessor_ifd = load(
    open("mlsupplychain/preprocessor_ifd/preprocessor_ifd.pkl", "rb")
)
preprocessor_otd = load(
    open("mlsupplychain/preprocessor_otd/preprocessor_otd.pkl", "rb")
)
model_ifd = keras.models.load_model("mlsupplychain/model_ifd")
model_otd = keras.models.load_model("mlsupplychain/model_otd")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")

    payload = {
        k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()
    }
    customer_id = int(next(iter(payload["customer_id"])))
    customer_name = df[df["customer_id"] == customer_id]["customer_name"].iloc[0]
    city = df[df["customer_id"] == customer_id]["city"].iloc[0]
    product_name = " ".join(
        s.capitalize() if i else s.upper()
        for i, s in enumerate(payload["product_name"][0].split())
    )
    category = df[df["product_name"] == product_name]["category"].iloc[0]
    sub_category = df[df["product_name"] == product_name]["sub_category"].iloc[0]
    try:
        order_qty = int(next(iter(payload["order_qty"])))
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input for order quantity.",
        )
    infull_target = df[df["customer_id"] == customer_id]["infull_target%"].iloc[0]
    ontime_target = df[df["customer_id"] == customer_id]["ontime_target%"].iloc[0]
    order_date = datetime.datetime.strptime(payload["order_date"][0], "%Y-%m-%d")
    week_no = order_date.isocalendar()[1]
    days_for_delivery = int(next(iter(payload["days_for_delivery"])))
    total_order_qty = order_qty
    month = order_date.month
    day = order_date.day
    dayofweek = order_date.weekday()
    order_rate = 1

    X_ifd = pd.DataFrame(
        {
            "customer_id": [str(customer_id)],
            "customer_name": [customer_name],
            "city": [city],
            "product_name": [product_name],
            "category": [category],
            "sub_category": [sub_category],
            "order_qty": [order_qty**0.5],
            "infull_target%": [infull_target],
            "week_no": [week_no],
            "days_for_delivery": [days_for_delivery],
            "total_order_qty": [total_order_qty],
            "month": [month],
            "day": [day],
            "dayofweek": [dayofweek],
            "order_rate": [order_rate],
        }
    )
    logging.info(X_ifd)
    logging.info(X_ifd.info())

    X_otd = pd.DataFrame(
        {
            "customer_id": [str(customer_id)],
            "customer_name": [customer_name],
            "city": [city],
            "product_name": [product_name],
            "category": [category],
            "sub_category": [sub_category],
            "order_qty": [order_qty**0.5],
            "ontime_target%": [ontime_target],
            "week_no": [week_no],
            "days_for_delivery": [days_for_delivery],
            "total_order_qty": [total_order_qty],
            "month": [month],
            "day": [day],
            "dayofweek": [dayofweek],
            "order_rate": [order_rate],
        }
    )
    logging.info(X_otd)
    logging.info(X_otd.info())

    logging.info("*******Prediction*******")
    prediction_ifd = model_ifd.predict(preprocessor_ifd.transform(X_ifd))[0][0]
    prediction_otd = model_otd.predict(preprocessor_otd.transform(X_otd))[0][0]
    logging.info(f"prediction_ifd: {prediction_ifd}")
    logging.info(f"prediction_otd: {prediction_otd}")

    return func.HttpResponse(
        status_code=200,
        body=f"The probability of 'IN FULL DELIVERY' is {prediction_ifd * 100:.2f}% and 'ON TIME DELIVERY' is {prediction_otd * 100:.2f}%.",
    )
