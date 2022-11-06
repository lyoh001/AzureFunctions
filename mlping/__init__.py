import logging

import azure.functions as func
import requests


def main(mytimer: func.TimerRequest) -> None:
    logging.info("*******Starting main function*******")
    logging.info(
        requests.post(
            "https://white-water-02d382d10.2.azurestaticapps.net/api/mlmobileuurka",
            json={
                "Age": [""],
                "BS": [""],
                "BodyTemp": [""],
                "DiastolicBP": [""],
                "HeartRate": [""],
                "SystolicBP": [""],
            },
        )
    )
    logging.info(
        requests.post(
            "https://white-water-02d382d10.2.azurestaticapps.net/api/mlsupplychain",
            json={
                "customer_id": ["789203"],
                "days_for_delivery": ["1"],
                "order_date": ["2022-01-01"],
                "order_qty": ["10"],
                "product_name": ["AM Ghee 250"],
            },
        )
    )
