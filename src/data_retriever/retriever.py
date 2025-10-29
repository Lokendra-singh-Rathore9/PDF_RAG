import os
from operator import itemgetter
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


from logger import GLOBAL_LOGGER as log
from src.prompt.prompt_library import ChatPromptTemplate, PROMPT_REGISTRY
from langchain_groq import ChatGroq


from dotenv import load_dotenv

load_dotenv()



class ConversationalRAG:

    def __init__(self, session_id: str):
        try:
            self.session_id = session_id
            self.retriever = None

            # Load LLM and prompts once
            self.llm = ChatGroq(
                                model="openai/gpt-oss-20b",
                                temperature=0,
                                max_tokens=None,
                                reasoning_format="parsed",
                                timeout=None,
                                max_retries=2,
                                api_key=os.getenv("GROQ_API_KEY"),
                            )
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                "contextualize_question"
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                "context_qa"
            ]

            self.chain = None

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise e
        
    def load_retriever_from_faiss(
        self,
        index_path: str,
    ) -> None:
        """Load retriever from persisted FAISS index."""
        try:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # Load FAISS vector store
            vector_store = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )

            self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            print()
            self._build_lcel_chain()

            return self.retriever
        
            log.info("Retriever loaded from FAISS", index_path=index_path)
        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise e
        
    
    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise ValueError("LCEL chain is not built. Call _build_lcel_chain() first.")
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return str(answer)
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise e
        

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise ValueError("Retriever must be set before building LCEL chain.")

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise e