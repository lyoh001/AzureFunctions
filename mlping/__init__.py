import logging

import azure.functions as func
import requests


def main(mytimer: func.TimerRequest) -> None:
    logging.info("*******Starting main function*******")
    url = "https://orange-water-0538b1c10.2.azurestaticapps.net/"
    logging.info(
        requests.post(
            url=f"{url}mlhospitality",
            json={
                "City_Code": [""],
                "Accomodation_Type": [""],
                "Reco_Insurance_Type": [""],
                "Upper_Age": [""],
                "Lower_Age": [""],
                "Is_Spouse": [""],
                "Health Indicator": [""],
                "Holding_Policy_Duration": [""],
                "Holding_Policy_Type": [""],
                "Reco_Policy_Cat": [""],
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
                "order_qty": ["10"],
                "product_name": ["AM Ghee 250"],
            },
        )
    )
