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

preprocessor = load(open("mlhospitality/preprocessor/preprocessor.pkl", "rb"))
model = keras.models.load_model("mlhospitality/model")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")
    payload = pd.DataFrame(
        {k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()},
        dtype="object",
    )
    return func.HttpResponse(
        status_code=200,
        body=f"{model.predict(preprocessor.transform(payload))[0][0]:.2f}",
    )
