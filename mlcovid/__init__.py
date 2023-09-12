import logging
import warnings

import azure.functions as func
import cv2
import numpy as np
import requests
import tflite_runtime.interpreter as tflite

warnings.filterwarnings("ignore")

model = tflite.Interpreter(model_path="mlcovid/model/model.tflite")
URL = (
    "https://raw.githubusercontent.com/lyoh001/AzureStaticWebApps/main/mlcovid/images/"
)

normal = [1, 2, 4, 7, 8, 10, 13, 16, 19, 20, 22, 25, 28]
covid = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
pneumonia = [5, 11, 14, 17, 23, 26, 29]


def predict_image(image_id):
    try:
        preprocessed_image = (
            np.array(
                cv2.imdecode(
                    np.asarray(
                        bytearray(requests.get(f"{URL}{image_id}.jpg").content),
                        dtype=np.uint8,
                    ),
                    -1,
                )
            ).reshape(1, 224, 224, -1)
            / 255.0
        )
    except ConnectionError as e:
        return func.HttpResponse(
            status_code=400,
            body=f"Connection Error: {e}",
        )
    model.allocate_tensors()
    input_index = model.get_input_details()[0]["index"]
    input_tensor = preprocessed_image.astype("float32")
    output_details = model.get_output_details()
    model.set_tensor(input_index, input_tensor)
    model.invoke()
    prediction = ["Normal", "Covid", "Pneumonia"][
        np.argmax(model.get_tensor(output_details[0]["index"])[0])
    ]
    return prediction


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
                body="Invalid input, please use a number between 1 and 30.",
            )
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input, please use a number between 1 and 30.",
        )

    actual = (
        "Normal"
        if image_id in normal
        else "Covid"
        if image_id in covid
        else "Pneumonia"
    )
    prediction = predict_image(image_id)
    logging.info(f"Actual: {actual}  |  Prediction: {prediction}")
    logging.info(
        f"The AI predicts that the patient in the x-ray image is {'not infected with any disease' if prediction == 'Normal' else f'infected with {prediction}'}. Please note that the accuracy of the model will depend on the quality of the x-ray image and the size of the training dataset. If you are concerned about the accuracy of the prediction, you should consult with a medical professional."
    )

    return func.HttpResponse(
        status_code=200,
        body=f"The AI predicts that the patient in the x-ray image is {'not infected with any disease' if prediction == 'Normal' else f'infected with {prediction}'}. Please note that the accuracy of the model will depend on the quality of the x-ray image and the size of the training dataset. If you are concerned about the accuracy of the prediction, you should consult with a medical professional.",
    )
