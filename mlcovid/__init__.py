import logging
import os
import warnings

import azure.functions as func
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model = keras.models.load_model("mlcovid/model")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")
    payload = {
        k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()
    }
    try:
        image_id = int(next(iter(payload["image_id"])))
        if not 1 <= image_id <= 30:
            return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 1 and 30.",
        )
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 1 and 30.",
        )

    normal = [1, 2, 4, 7, 8, 10, 13, 16, 19, 20, 22, 25, 28]
    covid = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    pneumonia = [5, 11, 14, 17, 23, 26, 29]
    actual = "Normal" if image_id in normal else "Covid" if image_id in covid else "Pneumonia"
    preprocessed_image = np.array(cv2.resize(cv2.imread(os.path.join("mlcovid/images", f"{image_id}.jpg"))[:,:,::-1], (224, 224))).reshape(1, 224, 224, -1) / 255.0
    prediction = ["Normal", "Covid", "Pneumonia"][np.argmax(model.predict(preprocessed_image), axis=1)[0]]
    logging.info(f"Prediction: {prediction}\tActual: {actual}")

    return func.HttpResponse(
        status_code=200,
        body=f"Prediction: {prediction}\tActual: {actual}"
    )