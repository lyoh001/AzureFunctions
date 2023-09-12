import logging
import os

import azure.functions as func
import requests


def get_links():
    return [
        "mlcloudaudit",
        "mlcoffeeplantdiseases",
        "mlcovid",
        "mlmaternalmortality",
        "mlsupplychain",
        "mltelecom",
    ]


def get_bodies():
    return {
        "mlcloudaudit": {"Date": ["2023-01"]},
        "mlcoffeeplantdiseases": {"image_id": ["1"]},
        "mlcovid": {"image_id": ["1"]},
        "mlmaternalmortality": {
            "Age": ["10"],
            "BS": ["6"],
            "BodyTemp": ["103"],
            "DiastolicBP": ["49"],
            "HeartRate": ["90"],
            "SystolicBP": ["70"],
        },
        "mlsupplychain": {
            "customer_id": ["789203"],
            "days_for_delivery": ["1"],
            "order_date": ["2022-01-01"],
            "order_qty": ["20"],
            "product_name": ["AM Ghee 250"],
        },
        "mltelecom": {
            "city_name": ["Gurgaon"],
            "time_period": ["2022-11"],
            "data_size": ["30"],
            "valid_period": ["180"],
        },
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    links = get_links()[3:4]
    bodies = get_bodies()
    for link in links:
        logging.info(
            requests.post(
                url=f"{os.environ['URL']}/api/{link}",
                json=bodies[link],
            )
        )

    return func.HttpResponse(status_code=200, body=f"Ready.")
