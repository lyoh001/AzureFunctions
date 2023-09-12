import datetime
import logging
import warnings
from pickle import load

import azure.functions as func
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite

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
model_ifd = tflite.Interpreter(model_path="mlsupplychain/model_ifd/model.tflite")
model_otd = tflite.Interpreter(model_path="mlsupplychain/model_otd/model.tflite")


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
    model_ifd.allocate_tensors()
    input_index = model_ifd.get_input_details()[0]["index"]
    input_tensor = preprocessor_ifd.transform(X_ifd).astype("float32")
    output_details = model_ifd.get_output_details()
    model_ifd.set_tensor(input_index, input_tensor)
    model_ifd.invoke()
    prediction_ifd = model_ifd.get_tensor(output_details[0]["index"])[0][0]
    logging.info(f"prediction_ifd: {prediction_ifd}")

    model_otd.allocate_tensors()
    input_index = model_otd.get_input_details()[0]["index"]
    input_tensor = preprocessor_otd.transform(X_otd).astype("float32")
    output_details = model_otd.get_output_details()
    model_otd.set_tensor(input_index, input_tensor)
    model_otd.invoke()
    prediction_otd = model_otd.get_tensor(output_details[0]["index"])[0][0]
    logging.info(f"prediction_otd: {prediction_otd}")

    output = f"The AI model predicts that the probability of in-full delivery is {prediction_ifd * 100:.2f}% and the probability of on-time delivery is {prediction_otd * 100:.2f}%. These probabilities provide insights into the likelihood of successful and on-time delivery for the supply."
    logging.info(output)

    return func.HttpResponse(
        status_code=200,
        body=output,
    )
