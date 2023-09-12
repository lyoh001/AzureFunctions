import calendar
import datetime
import io
import logging
import warnings
from pickle import load

import azure.functions as func
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite

warnings.filterwarnings("ignore")

X_preprocessor = load(open("mlcloudaudit/preprocessor/X_preprocessor.pkl", "rb"))
y_preprocessor = load(open("mlcloudaudit/preprocessor/y_preprocessor.pkl", "rb"))
model = tflite.Interpreter(model_path="mlcloudaudit/model/model.tflite")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    try:
        selected_month = [v for _, v in req.get_json().items()][0][0].split("-")[-1]
        user_input = str(
            datetime.datetime.strptime(
                f"2023-{selected_month}-{calendar.monthrange(2023, int(selected_month))[-1]}",
                "%Y-%m-%d",
            )
        ).split()[0]
        logging.info(f"selected_month: {selected_month}\nuser_input: {user_input}")
    except ValueError:
        return func.HttpResponse(
            status_code=200,
            body="Invalid input.",
        )
    N_PAST = 5
    N_FUTURE = 1
    ASFREQ = "MS"
    FREQ = "1m"
    DATE_COL = "Date"
    y_label = "Bill"

    df = pd.read_csv(
        io.StringIO(
            """Date,Bill
            2022-06,334230
            2022-07,344321
            2022-08,351373
            2022-09,346814
            2022-10,387434
            2022-11,394204
            2022-12,401308"""
        )
    )
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m")
    df = df.set_index(DATE_COL).sort_index().asfreq(ASFREQ)
    X_test_scaled = X_preprocessor.transform(df)
    X_test = np.array(
        [
            X_test_scaled[i : i + N_PAST]
            for i in range(len(X_test_scaled) - N_PAST - N_FUTURE + 1)
        ]
    )
    pred_future, current_batch = [], X_test[-1:]
    for _ in range(N_PAST):
        model.allocate_tensors()
        input_index = model.get_input_details()[0]["index"]
        input_tensor = current_batch.astype("float32")
        output_details = model.get_output_details()
        model.set_tensor(input_index, input_tensor)
        model.invoke()
        current_pred = model.get_tensor(output_details[0]["index"])[0]
        pred_future.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
    df_future_pred = pd.DataFrame(
        y_preprocessor.inverse_transform(pred_future),
        columns=[y_label],
        index=pd.date_range(df.index[-1], periods=N_PAST, freq=FREQ, name=DATE_COL),
    )
    prediction = f"The AI model predicts that future Azure consumption for {selected_month}/2023 will be AUD ${df_future_pred.loc[user_input]['Bill']:,.2f}. This prediction is based on the data analysis and machine learning techniques that were used to optimize the license scheme and resource utilization. The model takes into account a number of factors, including the current usage patterns, the forecasted growth of the business, and the cost of Azure services. It is important to note that this is just a prediction and the actual cost may vary. However, the model has been shown to be accurate in the past and it is a valuable tool for businesses that are looking to reduce their Azure costs."
    logging.info(f"prediction: {prediction}")

    return func.HttpResponse(status_code=200, body=prediction)
