from dataclasses import dataclass
from typing import List

from langchain.chains import OpenAIModerationChain
from langchain.chat_models.base import BaseChatModel
from langchain.vectorstores import Pinecone
import pinecone

from langchain import LLMMathChain, FAISS, InMemoryDocstore, PromptTemplate, LLMChain
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, BaseChatMessageHistory
from langchain.utilities import BingSearchAPIWrapper
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from time import time

from langchain.vectorstores.base import VectorStoreRetriever

from custom_memory import ConversationLSTMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferWindowMemory
from datetime import datetime
import os

from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities import GoogleSearchAPIWrapper

from dotenv import load_dotenv

load_dotenv()

# import langchain
# langchain.debug = True


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

perfect_session_prompt = """Greet {name} and welcoming him back, congratulate him to have completed {n_sessions} training session, averaging {avg_score} points.
Compliment him for achieving a perfect score on the last training session. 
Finally ask him if there is anything else he like to talk about."""

mistaken_session_prompt = """Generate a welcome message for {name} composed of 4  phrases:
In the first phrase greet {name} and welcoming him back, congratulate him to have completed {n_sessions} training session, averaging {avg_score} points.

Then consider this:
Question: {question}
Correct answer: {correct_answer}

In the second phrase remind him that in the last training session {name} replied incorrectly to the above Question.
In the third phrase shortly explain him the correct answer. 
In the fourth phrase ask him a question to continue the conversation.

Do not respond with a numbered list ot bullet points."""


@dataclass
class BotPersonality:
    bot_name: str
    bot_personality_instruction: str


# def get_current_date():
#     return datetime.today().strftime('%Y-%m-%d')
class Chatbot:
    def __init__(self, username: str, bot_personality: BotPersonality,
                 llm: BaseChatModel = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")):
        self.username = username
        self.bot_personality: BotPersonality = bot_personality
        self.llm: BaseChatModel = llm
        self.tools: List[Tool] = None
        self.long_term_memory: VectorStoreRetriever = None
        self.short_term_memory: BaseChatMessageHistory = None
        self.st_memory_interactions: int = None
        self.LSTM_memory: ConversationLSTMemory = None
        self.agent: AgentExecutor = None

    def init_tools(self, tools: List[Tool] = None):
        if tools:
            self.tools = tools
        else:
            search = BingSearchAPIWrapper()
            search_tool = Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about the current state of the world. You should ask targeted questions"
            )
            llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)
            calc_tool = Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer mathematical questions. You can input only numerical expression"
            )
            self.tools = [search_tool, calc_tool]

    def init_long_term_memory(self, env: str, index_name: str, returned_interactions: int,
                              interactions_threshold: float):
        """
        Init Long Term Memory VectorStoreRetriever
        @param env: Pinecone env
        @param index_name: Pinecone index
        @param returned_interactions: n. of returned interactions between user and bot
        @param interactions_threshold: similarity threshold for returned interactions
        """
        pinecone.init(environment=env)
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
                               embedding=embeddings.embed_query,
                               text_key="text",
                               namespace=self.username,
                               )
        self.long_term_memory = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                                         search_kwargs=dict(k=returned_interactions,
                                                                            # search_type="mmr",
                                                                            # fetch_k=4,
                                                                            # lambda_mult=0.2,
                                                                            score_threshold=interactions_threshold
                                                                            ))

    def init_short_term_memory(self, ttl, returned_interactions):
        self.short_term_memory = RedisChatMessageHistory(url=os.getenv('REDIS_URL'), ttl=ttl, session_id=self.username)
        self.st_memory_interactions = returned_interactions

    # def combine_memories(self):
    #     if not self.long_term_memory or not self.short_term_memory:
    #         raise NotImplementedError
    #     self.LSTM_memory = ConversationLSTMemory(memory_key="memory",
    #                                              st_memory_interactions=self.st_memory_interactions,
    #                                              human_prefix=self.username,
    #                                              ai_prefix=self.bot_personality.bot_name,
    #                                              chat_memory=self.short_term_memory,
    #                                              long_term_retriever=self.long_term_memory,
    #                                              input_key="input",
    #                                              return_messages=True)

    def init_agent(self, agent_type: AgentType = AgentType.OPENAI_FUNCTIONS):
        if not self.tools or not self.llm or not self.long_term_memory or not self.short_term_memory:
            raise NotImplementedError
        self.LSTM_memory = ConversationLSTMemory(memory_key="memory",
                                                 st_memory_interactions=self.st_memory_interactions,
                                                 human_prefix=self.username,
                                                 ai_prefix=self.bot_personality.bot_name,
                                                 chat_memory=self.short_term_memory,
                                                 long_term_retriever=self.long_term_memory,
                                                 input_key="input",
                                                 return_messages=True)
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": SystemMessage(content=self.bot_personality.bot_personality_instruction)
        }
        self.agent = initialize_agent(self.tools,
                                      self.llm,
                                      agent=agent_type,
                                      # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                      agent_kwargs=agent_kwargs,
                                      memory=self.LSTM_memory,
                                      max_execution_time=10,
                                      verbose=True)

    def welcome_message(self, template: str, template_kwargs: dict) -> str:
        """
        Generate welcome message from template and kwargs
        @param template: string template
        @param template_kwargs: dict containing the template kwargs
        @return: welcome message
        """

        prompt = PromptTemplate(
            input_variables=list(template_kwargs.keys()),
            template=template,
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        message = chain.run(**template_kwargs)
        self.LSTM_memory.add_st_ai_message(message)
        return message

    def chat(self, message: str, moderation: bool = True):
        if not self.agent:
            raise NotImplementedError
        response = self.agent.run(message)
        if moderation:
            moderation_chain_error = OpenAIModerationChain(error=True)
            try:
                moderation_chain_error.run(response)
            except ValueError:
                return "Sorry"
        return response


# username = "aaa"
# steve_personality = BotPersonality("Steve", system_message)
# chatbot = Chatbot(username, steve_personality)
# chatbot.init_tools()
# chatbot.init_short_term_memory(ttl=900, returned_interactions=4)
# chatbot.init_long_term_memory(env="gcp-starter", index_name='langchain-demo1',
#                               returned_interactions=4, interactions_threshold=0.4)
# chatbot.init_agent()


def get_agent(namespace):
    print(f"namespace: {namespace}")
    # llm = PromptLayerChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", streaming=True)

    ##########################################

    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    search = BingSearchAPIWrapper()
    # search = DuckDuckGoSearchRun()
    # search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # search = GoogleSearchAPIWrapper()

    search_tool = Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about the current state of the world. You should ask targeted questions"
    )

    calc = Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer mathematical questions, you can input only numerical expression"
    )

    # cur_date = Tool(
    #     name="Date",
    #     func=get_current_date,
    #     description="useful for when you need to know the current date"
    # )

    tools = [search_tool, calc]

    ##########################################
    pinecone.init(environment="eu-west4-gcp")
    # pinecone.init(environment="gcp-starter")

    index_name = 'chainlit_demo'
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
                           embedding=embeddings.embed_query,
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
    memory = ConversationLSTMemory(memory_key="memory",
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
                                   agent=AgentType.OPENAI_FUNCTIONS,
                                   # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                   agent_kwargs=agent_kwargs,
                                   memory=memory,
                                   max_execution_time=10,
                                   verbose=True)

    return agent_chain

# agent_chain  = get_agent('test1')
#
# start = time()
# print(agent_chain.run(input="what is the summed age of the actors brad pit and leonardo di caprio raised to the power of 5?"))
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
