import logging

import azure.functions as func
import requests


def main(mytimer: func.TimerRequest) -> None:
    logging.info("*******Starting main function*******")
    url = "https://thankful-pond-0298bdb10.2.azurestaticapps.net/api/"
    logging.info(
        requests.post(
            url=f"{url}mltelecom",
            json={
                "city_name": ["Gurgaon"],
                "time_period": ["2022-11"],
                "data_size": ["30"],
                "valid_period": ["180"],
            },
        )
    )
    logging.info(
        requests.post(
            url=f"{url}mlmobileuurka",
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
            url=f"{url}mlsupplychain",
            json={
                "customer_id": ["789203"],
                "days_for_delivery": ["1"],
                "order_date": ["2022-01-01"],
                "order_qty": ["1"],
                "product_name": ["AM Ghee 250"],
            },
        )
    )
