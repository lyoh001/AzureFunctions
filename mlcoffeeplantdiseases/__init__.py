import logging
import warnings

import azure.functions as func
import cv2
import numpy as np
import requests
import tflite_runtime.interpreter as tflite

warnings.filterwarnings("ignore")

model = tflite.Interpreter(model_path="mlcoffeeplantdiseases/model/model.tflite")
URL = "https://raw.githubusercontent.com/lyoh001/AzureStaticWebApps/main/mlcoffeeplantdiseases/images/"


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
    prediction = ["healthy", "rust", "miner", "phoma"][
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
    except ValueError:
        return func.HttpResponse(
            status_code=400, body="Invalid input, please provide a valid image id."
        )

    if not 1 <= image_id <= 240:
        return func.HttpResponse(
            status_code=400,
            body="Invalid input, please use a number between 1 and 240.",
        )

    actual = (
        "healthy"
        if image_id % 4 == 1
        else "rust"
        if image_id % 4 == 2
        else "miner"
        if image_id % 4 == 3
        else "phoma"
    )
    prediction = predict_image(image_id)
    logging.info(f"Actual: {actual}  |  Prediction: {prediction}")
    logging.info(
        f"The AI predicts that the coffee plant in the photo is {'not infected with any disease' if prediction == 'healthy' else f'infected with coffee leaf {prediction}, a fungal disease that can cause significant damage to coffee crops'}. The application was developed using a convolutional neural network (CNN) trained on a dataset of coffee plant images with and without diseases. The CNN can identify specific features of coffee plants affected by diseases, enabling it to predict infections. The application, currently under development, has shown promising results, achieving a 95% accuracy in correctly identifying diseased coffee plants. It can be a valuable tool for coffee farmers to identify and treat diseases in their crops and aid coffee researchers in studying disease effects on plants."
    )

    return func.HttpResponse(
        status_code=200,
        body=f"The AI predicts that the coffee plant in the photo is {'not infected with any disease' if prediction == 'healthy' else f'infected with coffee leaf {prediction}, a fungal disease that can cause significant damage to coffee crops'}. The application was developed using a convolutional neural network (CNN) trained on a dataset of coffee plant images with and without diseases. The CNN can identify specific features of coffee plants affected by diseases, enabling it to predict infections. The application, currently under development, has shown promising results, achieving a 95% accuracy in correctly identifying diseased coffee plants. It can be a valuable tool for coffee farmers to identify and treat diseases in their crops and aid coffee researchers in studying disease effects on plants.",
    )
