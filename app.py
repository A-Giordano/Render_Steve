import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from Function_pinecone import get_agent
import uuid


@cl.langchain_factory(use_async=False)
def factory():
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    return get_agent(namespace=str(uuid.uuid4()))
