from dotenv import load_dotenv
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

def welcome_message():
    llm = ChatOpenAI(temperature=0.5)

    prompt = PromptTemplate(
        input_variables=["name", "n_sessions", "avg_score", "question", "correct_answer"],
        template=mistaken_session_prompt,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    # return chain.run(name="Andrea", n_sessions=5, avg_score=700, question="what colour is the sky?", correct_answer= "blue")
    return chain.run(**{"name": "Andrea", "n_sessions": 5, "avg_score": 700, "question": "what colour is the sky?", "correct_answer": "blue"})

def welcome_message2(template: str, template_kwargs: dict):
    llm = ChatOpenAI(temperature=0.5)

    prompt = PromptTemplate(
        input_variables=list(template_kwargs.keys()),
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    message = chain.run(**template_kwargs)

    return message

load_dotenv()
a = welcome_message2(mistaken_session_prompt,
                     {"name": "Andrea", "n_sessions": 5, "avg_score": 700, "question": "what colour is the sky?", "correct_answer": "blue"})
pass


