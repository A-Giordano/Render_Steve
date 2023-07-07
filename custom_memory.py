from typing import Any, Dict, List, Optional, Union

from langchain.memory import ChatMessageHistory
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import root_validator, Field
from langchain.schema import Document

from langchain.memory.chat_memory import BaseChatMemory, BaseMemory
from langchain.memory.utils import get_prompt_input_key
from langchain.schema import get_buffer_string
from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
)
from datetime import date


class ConversationLTSTMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""
    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    long_term_retriever: VectorStoreRetriever = Field(exclude=True)
    # short_term_memory: BaseMemory
    # return_docs: bool = False
    k: int = 2

    # @property
    # def buffer(self) -> Any:
    #     """String buffer of memory."""
    #     short_term_memory = self.chat_memory.messages
    #     long_term_memory = self.load_lt_memory_variables()
    #     long_term_memory.extend(short_term_memory)
    #     if self.return_messages:
    #         return long_term_memory
    #
    #     else:
    #         return get_buffer_string(
    #             long_term_memory,
    #             human_prefix=self.human_prefix,
    #             ai_prefix=self.ai_prefix,
    #         )

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        short_term_memory = self.chat_memory.messages[-self.k * 2:] if self.k > 0 else []
        long_term_memory = self.load_lt_memory_variables(inputs)
        total_memory = [AIMessage(content="Hi, I'm Steve Jobs, the iconic co-founder of Apple.")]
        total_memory.extend(long_term_memory + short_term_memory)
        # long_term_memory.extend(short_term_memory)
        print([mex.content for mex in total_memory])
        # Filter messages to have only the uniques
        unique_contents = set()
        filtered_messages = [msg for msg in total_memory if
                             msg.content not in unique_contents
                             and not unique_contents.add(msg.content)]

        print([mex.content for mex in filtered_messages])
        print(*[mex.content for mex in filtered_messages], sep="\n\n")
        if self.return_messages:
            return {self.memory_key: filtered_messages}

        else:
            return {self.memory_key: get_buffer_string(
                filtered_messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        # short term memory
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)
        # long term memory
        documents = self._form_documents(inputs, outputs)
        self.long_term_retriever.add_documents(documents)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()

    def load_lt_memory_variables(
            self, inputs: Dict[str, Any]
    ) -> list:
        """Return history buffer."""
        input_key = self._get_prompt_input_key(inputs)
        query = inputs[input_key]
        docs = self.long_term_retriever.get_relevant_documents(query)
        # result: Union[List[Document], str]
        # if self.return_messages:
        #     result = [doc.page_content for doc in docs]
        # # elif not self.return_docs and not self.return_messages:
        # #     result = "\n".join([doc.page_content for doc in docs])
        # else:
        #     # result = docs
        #     result = "\n".join([doc.page_content for doc in docs])
        result = [self.parse_retriever(doc) for doc in docs]
        result = [item for sublist in result for item in sublist]

        return result

    @staticmethod
    def parse_retriever(doc):
        parts = doc.page_content.split("input:")[1].split("output:")
        user_input = parts[0].strip()
        output = parts[1].strip()

        return HumanMessage(content=user_input), AIMessage(content=output)

    def _form_documents(
            self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content,
                         metadats={"date": date.today()})]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key
