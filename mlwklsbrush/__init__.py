import json
import logging
import os

import azure.functions as func
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def brush(data: dict) -> str:
    llm = AzureChatOpenAI(
        deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
        model_name=os.environ["OPENAI_MODEL_NAME"],
        temperature=1,
    )
    messages = [
        SystemMessage(
            content="You are a Korean teacher at a primary school in Australia, and your task is to complete the end-of-year student report card. Here is a sample report that you can refer to: ```Jenny has made great progress this semester! She is a kind and gentle member of our class. She has grown in self-confidence throughout the semester. Jenny contributes to class discussions, is attentive, and always follows instructions. She confidently seeks clarification if she does not understand what is expected. She has made connections and friendships with her peers. She can work in groups and independently. Overall, Jenny has made a great start to Grade 1! Keep it up, Jenny!```"
        ),
        HumanMessage(
            content=f"Write a concise overall comment for a student report card based on the following information. ```{json.dumps(data)}```  Follow these three instructions.\n 1. Use British English spelling.\n 2. Do not use new lines.\n 3. Aim to hit over 1300 characters, but do not exceed 1500 characters in total."
        ),
    ]
    return llm(messages).content


def get_rating(rating):
    return {
        "0": "Very Low",
        "1": "Needs Attention",
        "2": "Satisfactory",
        "3": "High",
        "4": "Excellent",
    }.get(rating, "Unknown")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting Brush function*******")
    try:
        user_input = req.get_json()
        data = {
            "student_name": user_input["studentName"],
            "grade": f"{grade}" if (grade := int(user_input["grade"])) else "Prep",
            "attendance": get_rating(user_input["attendance"]),
            "behaviour": get_rating(user_input["behaviour"]),
            "effort": get_rating(user_input["effort"]),
            "communication_skills": " and ".join(
                f"{skill.split(']')[1]}" for skill in user_input["communicationSkills"]
            ),
            "understanding_skills": " and ".join(
                f"{skill.split(']')[1]}" for skill in user_input["understandingSkills"]
            ),
        }
        return func.HttpResponse(
            json.dumps({"updatedComment": brush(data)}),
            status_code=200,
            mimetype="application/json",
        )

    except ValueError:
        return func.HttpResponse(
            "Invalid input, please provide a valid user_input.", status_code=400
        )
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
