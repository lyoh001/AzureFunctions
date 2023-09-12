import logging
import os

import azure.functions as func
from langchain import (FewShotPromptTemplate, HuggingFaceHub, LLMChain,
                       PromptTemplate)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting main function*******")
    try:
        question = next(
            iter(
                {
                    k: None if next(iter(v)) == "" else v
                    for k, v in req.get_json().items()
                }["text"]
            )
        )
        logging.info(f"Parsed Text: {question}")

    except ValueError:
        return func.HttpResponse(
            status_code=400, body="Invalid input, please provide a valid text."
        )

    HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]
    CLASSIFICATION = 1
    PREFIX = [
        "Your task is to provide accurate and detailed answers to the following question.",
        "Your task is to perform a multi-class classification on natural disaster tweets. This involves analyzing the text given in the question and assigning it to one of several categories. If the tweet is not disaster related, just return 'not a disaster'. Here are some examples.",
    ][CLASSIFICATION]
    EXAMPLES = [
        [],
        [
            {
                "question": "Just witnessed a massive landslide near my town. Praying for the safety of everyone affected.",
                "answer": "landslide",
            },
            {
                "question": "I can't believe the devastation caused by the hurricane. So many homes and businesses destroyed.",
                "answer": "hurricane",
            },
            {
                "question": "The flood has submerged the entire city. We are trapped on the roof of our building. Please send help.",
                "answer": "flood",
            },
            {
                "question": "The earthquake was so strong it shook my whole house. I hope there are no casualties.",
                "answer": "earthquake",
            },
            {
                "question": "The wildfire is getting closer and closer to our neighborhood. We are packing our bags and evacuating.",
                "answer": "wildfire",
            },
            {
                "question": "I love coffee.",
                "answer": "not a disaster",
            },
        ],
    ][CLASSIFICATION]
    MODEL_NAME = [
        "tiiuae/falcon-7b",
        "tiiuae/falcon-7b-instruct",
    ][-1]
    llm = HuggingFaceHub(
        repo_id=MODEL_NAME,
        model_kwargs={
            "max_new_tokens": [200, 3][CLASSIFICATION],
            "temperature": 0.1,
            "top_p": 0.7,
        },
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    )
    logging.info(f"The model has been constructed: {llm}")

    template = """\nquestion: {question}\nanswer: {answer}"""
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"], template=template
    )
    prefix = f"{PREFIX}"
    suffix = """\nquestion: {question}\nanswer: """.strip()
    prompt_template = FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["question"],
        example_separator="\n",
    )
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    logging.info(prompt_template.format(question=question))
    output = llm_chain.run({"question": question})
    logging.info(f"output: {output}")

    return func.HttpResponse(
        status_code=200,
        body=f"The LLM model predicts that the given tweet is related to a {output}.",
    )
