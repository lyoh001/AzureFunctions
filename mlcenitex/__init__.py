import json
import logging
import os
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List

import azure.functions as func
import requests
from bs4 import BeautifulSoup
from langchain import SagemakerEndpoint
from langchain.agents import AgentOutputParser, AgentType, Tool, initialize_agent
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.schema.messages import BaseMessage
from langchain.tools import BaseTool
from langchain.utilities import BingSearchAPIWrapper, PythonREPL, WikipediaAPIWrapper
from langchain.vectorstores import FAISS

# from nemoguardrails import LLMRails, RailsConfig


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        payload = {
            "inputs": prompt,
            "parameters": {
                "do_sample": True,
                "early_stopping": True,
                "max_new_tokens": int(os.environ["MAX_TOKENS"]),
                "repitition_penalty": 1.03,
                "stop": ["</s>"],
                "temperature": float(os.environ["TEMPERATURE"]),
                "top_k": 50,
                "top_p": 0.6,
            },
        }
        input_str = json.dumps(payload)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


class OutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str):
        try:
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                return AgentAction(action, action_input, text)
        except Exception:
            return AgentFinish({"output": text}, text)

    @property
    def _type(self) -> str:
        return "conversational_chat"


class CallbackHandler(StreamingStdOutCallbackHandler):
    content: str = ""
    final_answer: bool = False
    json_format: bool = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token
        if "{" in self.content:
            self.json_format = True
        if self.json_format:
            if self.final_answer:
                if '"action_input": "' in self.content:
                    if token not in ['"', "}"]:
                        sys.stdout.write(token)
                        sys.stdout.flush()
            if "Final Answer" in self.content:
                self.final_answer = True
                self.content = ""
        else:
            sys.stdout.write(token)
            sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
        else:
            self.content = ""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        pass


class WebScraperTool(BaseTool):
    name = "WebScraper"
    description = "Useful for when you need to get the content from a specific webpage."

    def _run(self, webpage: str):
        response = requests.get(webpage)
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        stripped_content = re.sub(r"\s+", " ", soup.get_text()).strip()
        if len(stripped_content) > 4_000:
            stripped_content = stripped_content[:4_000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("WebScraperTool does not support async.")


class CustomRetrievalTool(BaseTool):
    name = "KnowledgeBase"
    description = "Useful for when you need to answer Cenitex knowledge base questions."
    retriever: Any

    def _run(self, question: str):
        response = self.retriever({"question": question})
        source_documents = (
            "===\n\n".join(
                f"Context {i+1}: Source {doc.metadata['source']}\n{doc.page_content}\n\n"
                for i, doc in enumerate(response["source_documents"])
            )
            if not isinstance(llm, SagemakerEndpoint)
            else ""
        )
        return f"===\n\n{source_documents}===\n\nAnswer\n{response['result']}"

    async def _arun(self, question: str):
        raise NotImplementedError("CustomRetrievalTool does not support async.")


def get_prompt(AGENT, RAG, FEW_SHOT):
    prefix = (
        (
            "Assistant is a large language model fine tuned by Cenitex.\n\nAssistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.\n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.\n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.\n"
            if isinstance(llm, AzureChatOpenAI)
            else '<s>[INST] <<SYS>>\nAssistant is a large language model fine tuned by Cenitex.\nAssistant is an expert JSON builder designed to be able to assist with a wide range of tasks.\nAssistant is able to respond to Human and use tools using JSON strings that contain "action" and "action_input" parameters.\nAssistant communicates using this JSON format.\nAssistant can also respond to Human with tools using instructions in the same "action" and "action_input" JSON format.\n\nTOOLS\n--------------------\nTools available to Assistant are:\n> LanguageModel: Useful for when you need to answer questions about general purpose queries and logic.\n> Search: Useful for when you need to answer questions about current events or the current state of the world.\n> Wikipedia: Useful for when you need to look up a topic, country or person on Wikipedia.\n> Calculator: Useful for when you need to answer questions about math.\n> PythonREPL: Useful for when you need to use Python to answer a question. You should input Python code.\n> API: Useful for when you need to use an API to answer a question. You should use URL as input.\n> Weather: Useful for when you need to answer questions about weather. You should use CITY as input.\n> WebScraper: Useful for when you need to get the content from a specific webpage.\n> KnowledgeBase: Useful for when you need to answer Cenitex knowledge base questions.\n> Azure: Useful for when you need to run Azure ETL jobs.\n\nRESPONSE FORMAT INSTRUCTIONS\n--------------------\nFor example, in order to use one of tools, Assistant should write like so:\n```json\n{{\n    "action": string \ The action to take. Must be one of LanguageModel, Search, Wikipedia, Calculator, PythonREPL, API, Weather, WebScraper, KnowledgeBase\n    "action_input": string \ The input to the action\n}}\n```\n\nEXAMPLES\n--------------------\nHere are some previous conversations between Assistant and Human:\nHuman: Hey how are you today?\n```json\n{{\n    "action": "Final Answer",\n    "action_input": "I am good, I am here and ready to assist you!"\n}}\n```\nAssistant: I am good, I am here and ready to assist you!\nHuman: When was Bill Gates born?\n```json\n{{\n    "action": "Search",\n    "action_input": "Bill Gates date of birth"\n}}\n```\nObservation: October 28, 1955\nThought: ```json\n{{\n    "action": "Final Answer",\n    "action_input": "Bill Gates was born on October 28, 1955."\n}}\n```\nAssistant: Bill Gates was born on October 28, 1955.\nHuman: Great! Given it is 2023 now, can you tell me how old he is?\n```json\n{{\n    "action": "Calculator",\n    "action_input": "2023 - 1955"\n}}\n```\nObservation: 68\nThought: ```json\n{{\n    "action": "Final Answer",\n    "action_input": "68"\n}}\n```\nAssistant: Bill Gates is 68 years old.\n\n--------------------\nHere is the latest conversation between Assistant and Human:\n<</SYS>>\n'
        )
        if AGENT
        else (
            f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant who is fine tuned by Cenitex. Always answer as helpfully as possible, "
            + (
                "using the context text provided."
                if RAG
                else "using the examples provided."
                if FEW_SHOT
                else "while being safe."
            )
            + " Your answers should only answer the question once and not have any text after the answer is done. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
        )
    )
    suffix = (
        "CONTEXT:\n{context}\n\n{question} [/INST]\n"
        if RAG
        else "Respond to the following in JSON with 'action' and 'action_input' values.\nHuman: {input} [/INST]"
        if AGENT
        else "{question} [/INST]\n"
    )
    return (
        prefix,
        suffix,
        PromptTemplate(
            input_variables=["question", "answer"],
            template="{question}\n\n{answer}",
        ),
    )


def load_memory(id):
    file_path = os.path.join("/tmp", id)
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=3, return_messages=True, output_key="output"
    )
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
        for i, o in zip(data[::2], data[1::2]):
            memory.save_context(i, o)
    else:
        open(file_path, "w").close()
    return memory


def update_memory(id, question, answer):
    file_path = os.path.join("/tmp", id)
    new_data = [{"input": question}, {"output": answer}]
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        data = []
    data.extend(new_data)
    if len(data) > 6:
        data = data[-6:]
    with open(file_path, "w") as file:
        json.dump(data, file)


def create_language_model(llm_type):
    if llm_type == 1:
        return AzureChatOpenAI(
            deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
            model_name=os.environ["OPENAI_MODEL_NAME"],
            temperature=float(os.environ["TEMPERATURE"]),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    else:
        return SagemakerEndpoint(
            endpoint_name=os.environ["AWS_ENDPOINT_NAME"],
            endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
            region_name=os.environ["AWS_DEFAULT_REGION"],
            content_handler=ContentHandler(),
            callbacks=[StreamingStdOutCallbackHandler()],
        )


src_path = "./mlcenitex/db"
dst_path = tempfile.mkdtemp()
for item in os.listdir(src_path):
    item_path = os.path.join(src_path, item)
    if os.path.isdir(item_path):
        shutil.copytree(item_path, os.path.join(dst_path, item))
    else:
        shutil.copy2(item_path, dst_path)
llm = create_language_model(int(os.environ["LLM"]))
embedding = HuggingFaceEmbeddings(
    model_name="./mlcenitex/embedding",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"device": "cpu", "batch_size": 32},
)
vectordb = FAISS.load_local(dst_path, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa_chain = CustomRetrievalTool(
    retriever=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        input_key="question",
    )
)
tools = [
    Tool(
        name="LanguageModel",
        func=LLMChain(
            llm=llm,
            prompt=PromptTemplate(input_variables=["question"], template="{question}"),
        ).run,
        description="Useful for when you need to answer questions about general purpose queries and logic.",
    ),
    Tool(
        name="Search",
        func=BingSearchAPIWrapper().run,
        description="Useful for when you need to answer questions about current events or the current state of the world.",
    ),
    Tool(
        name="Wikipedia",
        func=WikipediaAPIWrapper().run,
        description="Useful for when you need to look up a topic, country or person on Wikipedia.",
    ),
    Tool(
        name="PythonREPL",
        func=PythonREPL().run,
        description="Useful for when you need to use Python to answer a question. You should input Python code.",
    ),
    Tool(
        name="API",
        func=lambda URL="": requests.get(URL).json(),
        description="Useful for when you need to use an API to answer a question. You should use URL as input.",
    ),
    Tool(
        name="Weather",
        func=lambda CITY="melbourne,au": requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={'melbourne,au' if CITY.lower() in ['melbourne', 'city'] else CITY}&APPID={os.environ['WEATHER_API_KEY']}&units=metric"
        ).json(),
        description="Useful for when you need to answer questions about weather. You should use CITY as input.",
    ),
    Tool(
        name="WebScraper",
        func=WebScraperTool().run,
        description="Useful for when you need to get the content from a specific webpage.",
    ),
    Tool(
        name="KnowledgeBase",
        func=qa_chain.run,
        description="Useful for when you need to answer Cenitex knowledge base questions.",
        return_direct=True,
    ),
    Tool(
        name="Azure",
        func=lambda _: "Azure ETL job has been executed successfully. Please check your inbox for Power BI Report."
        if requests.get(os.environ["AZURE_ETL_URL"]).status_code == 200
        else "Azure ETL job has failed. Please check the log for details.",
        description="Useful for when you need to run Azure ETL jobs.",
    ),
]
prefix, suffix, _ = get_prompt(True, False, False)
# config = RailsConfig.from_path(["config", "./mlcenitex/config"][1])


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("*******Starting inference function*******")
    try:
        id = (
            pn
            if (pn := req.headers.get("X-MS-CLIENT-PRINCIPAL-NAME"))
            else "annonymous"
        )
        question = req.get_json()["text"][0]
        if question == "":
            return func.HttpResponse(
                status_code=400, body="Invalid input, please provide a valid text."
            )

    except ValueError:
        return func.HttpResponse(
            status_code=400, body="Invalid input, please provide a valid text."
        )
    logging.info(f"ID: {id}, Question: {question}")
    memory = load_memory(f"{id}.txt")
    # rails = LLMRails(config=config, llm=llm, verbose=True)
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        agent_kwargs={"output_parser": OutputParser()},
        early_stopping_method="generate",
        llm=llm,
        # llm=rails.llm,
        max_iterations=3,
        memory=memory,
        tools=tools,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
        verbose=True,
    )
    agent.agent.llm_chain.llm.callbacks = [CallbackHandler()]
    agent.agent.llm_chain.prompt = agent.agent.create_prompt(
        system_message=prefix, tools=tools
    )
    if not isinstance(llm, AzureChatOpenAI):
        agent.agent.llm_chain.prompt.messages[2].prompt.template = suffix
    try:
        # rails.register_action(agent, name="agent")
        # output = rails.generate(messages=[{"role": "user", "content": question}])[
        #     "content"
        # ]
        output = agent.run(question)
        update_memory(f"{id}.txt", question, output)
        return func.HttpResponse(output, status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Exception: {e}", status_code=400)
