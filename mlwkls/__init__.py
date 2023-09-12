import logging
import os

import azure.functions as func
import openai


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting brush function*******")

    try:
        text = req.get_json()
        context = f"name: {text['name']}\ngrade: {text['grade']}\nAttendance: {text['studentAttendance']}\nskills: {' '. join(text['skills'])}"
        report = openai.ChatCompletion.create(
            engine=os.environ["OPENAI_DEPLOYMENT_NAME"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Can you write a concise student annual report?",
                },
                {
                    "role": "assistant",
                    "content": "Sure, please provide me with the information you would like to include in the student annual report.",
                },
                {
                    "role": "user",
                    "content": f"Sure, here are the student details: {context}\nThe report should only include 2~3 short comments.",
                },
            ],
        )["choices"][0]["message"]["content"]
        logging.info(f"context:\n{context}")
        logging.info(f"report:\n{report}")

    except ValueError:
        return func.HttpResponse(
            status_code=400, body="Invalid input, please provide a valid text."
        )
    return func.HttpResponse(report, status_code=200)