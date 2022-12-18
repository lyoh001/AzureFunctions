import logging
import os

import azure.functions as func
import requests


def main(mytimer: func.TimerRequest) -> None:
    logging.info("*******Starting main function*******")
    url = os.environ["URL"]
    logging.info(
        requests.post(
            url=f"{url}mlcloudaudit",
            json={"Date": ["2023-01"]},
        )
    )
    # logging.info(
    #     requests.post(
    #         url=f"{url}mltelecom",
    #         json={
    #             "city_name": ["Gurgaon"],
    #             "time_period": ["2022-11"],
    #             "data_size": ["30"],
    #             "valid_period": ["180"],
    #         },
    #     )
    # )
    # logging.info(
    #     requests.post(
    #         url=f"{url}mlmobileuurka",
    #         json={
    #             "Age": ["10"],
    #             "BS": ["6"],
    #             "BodyTemp": ["98"],
    #             "DiastolicBP": ["49"],
    #             "HeartRate": ["60"],
    #             "SystolicBP": ["70"],
    #         },
    #     )
    # )
    # logging.info(
    #     requests.post(
    #         url=f"{url}mlsupplychain",
    #         json={
    #             "customer_id": ["789203"],
    #             "days_for_delivery": ["1"],
    #             "order_date": ["2022-01-01"],
    #             "order_qty": ["1"],
    #             "product_name": ["AM Ghee 250"],
    #         },
    #     )
    # )
