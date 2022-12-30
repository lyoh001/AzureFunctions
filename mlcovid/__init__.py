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
        image_id = next(iter(payload["image_id"]))
        if not 1 <= int(image_id) <= 30:
            return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 1 and 30.",
        )
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use numbers between 1 and 30.",
        )

    # normal = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    # virus = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    # covid = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    # preprocessed_image = np.array(Image.open(os.path.join("mlcovid/images", f"{image_id}.jpg"))) / 255
    preprocessed_image = np.array(cv2.resize(cv2.imread(os.path.join("mlcovid/images", f"{image_id}.jpg"))[:,:,::-1], (224, 224))).reshape(1, 224, 224, -1) / 255.0
    prediction = model.predict(preprocessed_image)
    # prediction = np.argmax(model.predict(preprocessed_image), axis=1)
    logging.info(prediction)

    # prediction = model.predict(preprocessor.transform(X))[0][0]
    # logging.info(X)
    # logging.info(f"prediction: {prediction}")

    return func.HttpResponse(
        status_code=200,
        body="test"
        # body=f"{prediction}"
    )