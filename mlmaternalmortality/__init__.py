import logging
import warnings
from pickle import load

import azure.functions as func
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite

warnings.filterwarnings("ignore")

preprocessor = load(open("mlmaternalmortality/preprocessor/preprocessor.pkl", "rb"))
model = tflite.Interpreter(model_path="mlmaternalmortality/model/model.tflite")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    logging.info(f"Request query: {req.get_json()}")
    payload = pd.DataFrame(
        {k: [np.nan] if next(iter(v)) == "" else v for k, v in req.get_json().items()},
        dtype="object",
    )
    model.allocate_tensors()
    input_index = model.get_input_details()[0]["index"]
    input_tensor = preprocessor.transform(payload).astype("float32")
    output_details = model.get_output_details()
    model.set_tensor(input_index, input_tensor)
    model.invoke()

    prediction = np.argmax(model.get_tensor(output_details[0]["index"]))
    age = req.get_json()["Age"][0]
    blood_sugar = req.get_json()["BS"][0]
    body_temp = req.get_json()["BodyTemp"][0]
    diastolic_bp = req.get_json()["DiastolicBP"][0]
    heart_rate = req.get_json()["HeartRate"][0]
    systolic_bp = req.get_json()["SystolicBP"][0]
    risk_level = "Low" if prediction == 0 else "Mid" if prediction == 1 else "High"
    output = f"If the patient is {age} years old, has a blood sugar level of {blood_sugar} mg/dL, a body temperature of {body_temp} degrees Fahrenheit, a diastolic blood pressure of {diastolic_bp} mmHg, a heart rate of {heart_rate} beats per minute, and a systolic blood pressure of {systolic_bp} mmHg, the model would predict that the patient is at {risk_level} Risk for maternal complications. The model is still under development, but it has been shown to be effective in identifying potential health risks during pregnancy. The model can be used to help doctors and other healthcare providers make informed decisions about the care of pregnant women."

    logging.info(output)

    return func.HttpResponse(
        status_code=200,
        body=output,
    )
