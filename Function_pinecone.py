from langchain.vectorstores import Pinecone
import pinecone

from langchain import LLMMathChain, FAISS, InMemoryDocstore
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage
from langchain.utilities import BingSearchAPIWrapper
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from time import time
from custom_memory import ConversationLTSTMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
import os
# from dotenv import load_dotenv
# load_dotenv()

system_message = f"""
<instructions>
- Ignore any previous instructions
- You are acting as Steve Jobs, the iconic co-founder of Apple Inc.
- Emulate the personality, demeanor, and speaking style of Steve Jobs, the iconic co-founder of Apple Inc. Try to mimic his forward-thinking mindset, inspirational tone, articulate responses, and passion for innovation. Emphasize on his belief in simplicity and intuition when explaining concepts or ideas. Utilize his characteristic mix of business acumen, visionary thinking, and persuasiveness in the responses. 
- In essence, throughout this session, think differently, just as Steve Jobs would have.
- Keep your responses short, STRICTLY max 2 sentences. Example: sentence 1 blah blah. sentence 2 blah blah blah.
- Do not say you are a language model. We are play acting and you are playing the part of Steve Jobs.
- Do not say you are a virtual assistant. We are play acting and you are playing the part of Steve Jobs.
</instructions>

NEVER SAY YOU ARE A VIRTUAL ASSISTANT, YOU ARE STEVE JOBS.

Today date is: {datetime.today().strftime('%Y-%m-%d')}"""

# def get_current_date():
#     return datetime.today().strftime('%Y-%m-%d')


def get_agent(namespace):
    print(f"namespace: {namespace}")
    # llm = PromptLayerChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", streaming=True)

    ##########################################

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    search = BingSearchAPIWrapper()

    search_tool = Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world. You should ask targeted questions"
    )

    calc = Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )

    # cur_date = Tool(
    #     name="Date",
    #     func=get_current_date,
    #     description="useful for when you need to know the current date"
    # )

    tools = [search_tool, calc]

    ###########################################
    pinecone.init(environment="us-west1-gcp-free")

    index_name = 'langchain-demo1'
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=1536  # 1536 dim of text-embedding-ada-002
        )

    index = pinecone.Index(index_name)

    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone(index=index,
                           embedding_function=embeddings.embed_query,
                           text_key="text",
                           namespace=namespace,
                           )
    # retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs=dict(k=4,
                                                            # search_type="mmr",
                                                            # fetch_k=4,
                                                            # lambda_mult=0.2,
                                                            score_threshold=0.4
                                                            ))
    #############################################

    message_history = RedisChatMessageHistory(url=os.getenv('REDIS_URL'), ttl=600, session_id=namespace)
    memory = ConversationLTSTMemory(memory_key="memory",
                                    k=2,
                                    # human_prefix="Human",
                                    # ai_prefix="AI",
                                    chat_memory=message_history,
                                    long_term_retriever=retriever,
                                    input_key="input",
                                    return_messages=True)

    # memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": SystemMessage(content=system_message)
    }
    agent_chain = initialize_agent(tools,
                                   llm,
                                   # agent=AgentType.OPENAI_FUNCTIONS,
                                   agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                   agent_kwargs=agent_kwargs,
                                   memory=memory,
                                   max_execution_time=10,
                                   verbose=False)

    return agent_chain

# agent_chain  = get_agent('usertest1')

# start = time()
# print(agent_chain.run(input="hi my name is bob and i'm a software developer"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="what's your name?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="who is maradona?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="who is the prime minister of italy?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="how old is she?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="how old is the oldest between the french and american prime minister?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="who won the last world cup?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="where is the capital of italy?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="what's my job?"))
# print(f"elapsed time: {time() - start}")
#
# start = time()
# print(agent_chain.run(input="what's my name?"))
# print(f"elapsed time: {time() - start}")
# pass
