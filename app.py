import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from langchain.chains import OpenAIModerationChain

from Function_pinecone import get_agent
from welcome import welcome_message
import uuid
import time

# @cl.langchain_factory(use_async=False)
# def factory():
#     # prompt = PromptTemplate(template=template, input_variables=["question"])
#     # llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
#
#     return get_agent(namespace=str(uuid.uuid4()))

@cl.on_chat_start
def init():
    s = time.time()
    chain = get_agent(namespace=str(uuid.uuid4()))
    print(f"exec time: {time.time() - s}")
    welcome_msg = welcome_message()
    chain.memory.add_st_ai_message(welcome_msg)
    cl.user_session.set("chain", chain)
    cl.Message(content=welcome_message()).send()


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    res = chain.run(message)
    moderation_chain = OpenAIModerationChain(error=True)
    try:
        moderation_chain.run(res)
    except ValueError:
        res = "Sorry ValErr"
    # print(res)
    #
    # answer = res["result"]
    await cl.Message(content=res).send()
