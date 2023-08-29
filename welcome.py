from langchain.chat_models import PromptLayerChatOpenAI
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

In the second phrase remind him that in the last training session {name} replied incorrectly to the previous question.
In the third phrase shortly explain him the correct answer. 
In the fourth phrase ask him a question to continue the conversation."""

def welcome_message():
    llm = PromptLayerChatOpenAI(temperature=0.5)

    prompt = PromptTemplate(
        input_variables=["name", "n_session", "avg_score", "question", "correct_answer"],
        template=mistaken_session_prompt,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(name="Andrea", n_session=5, avg_score=700, question="what colour is the sky?", correct_answer= "blue")

