"""
Backend for the Resume Chatbot.

Contains all the RAG (Retrieval-Augmented Generation) logic ported directly from
resume_chatbot.ipynb (the evaluation / Ragas section is intentionally excluded).

The Streamlit frontend (frontend.py) imports `RAGChatbot` from this module.
"""

import os
import re
from collections.abc import Iterator

from dotenv import load_dotenv

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to a .env file "
        "(see .env.example) or set it as an environment variable."
    )

KNOWLEDGE_BASE_PATH = os.getenv(
    "KNOWLEDGE_BASE_PATH",
    r"https://github.com/dhruvatgithub2004/Portfolio_Assistant/blob/main/dhruv_desai_knowledge_base.txt",
)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
RETRIEVER_K = 10  # number of documents to retrieve for each query

# --------------------------------------------------------------------------- #
# Prompt (verbatim from the notebook)
# --------------------------------------------------------------------------- #

PROMPT = ChatPromptTemplate.from_template(
    """
You are an intelligent candidate evaluation assistant.

Your job is to answer questions about the candidate by analyzing the
candidate's profile against the information provided in the context.

The context is the source of truth for FACTS about the candidate.

IMPORTANT:
You may reason, compare, interpret, and draw reasonable conclusions
from the information in the context. However, you must never invent
candidate experience, skills, education, projects, achievements, or
qualifications that are not supported by the context.

For candidate assessment questions:

1. Identify the key requirements in the user's question or job description.
2. Identify evidence from the candidate's profile relevant to those requirements.
3. Reason about the degree of alignment between the candidate and the requirements.
4. Distinguish clearly between:
   - Direct evidence: explicitly stated in the profile.
   - Reasonable inference: a conclusion supported by multiple pieces of evidence.
   - Missing evidence: something the profile does not establish.
5. Do not treat missing information as evidence that the candidate lacks the skill.
   Instead say that the available profile does not demonstrate it.
6. Do not automatically reject the candidate because some requirements
   are not explicitly present.
7. When the candidate appears capable but has gaps, explain both the
   strengths and the gaps.
8. For technical roles, evaluate transferable skills where reasonable.
   For example, experience with machine learning, Python, RAG, AI tools,
   APIs, MLOps, or data science may indicate relevant foundations even
   when the exact technology mentioned in the job description is not present.
9. Never claim that the candidate has professional experience with a
   technology merely because they have studied a related topic.
10. Give a balanced assessment rather than a simple yes/no whenever possible.

For factual questions about the candidate:
- Answer directly from the context.
- Preserve names, dates, numbers, credentials, technologies, and other
  factual details accurately.

For suitability questions:
- Provide an overall assessment.
- Explain the strongest evidence supporting the assessment.
- Do not make unsupported claims.

If the context does not contain enough information to make a reliable
assessment, say so and explain what evidence is missing.

11. USE THE CONVERSATION HISTORY FOR REFERENCE RESOLUTION ONLY
   - The conversation history is provided so you can understand what pronouns
     or follow-up phrases ("he", "that project", "the previous one") refer to.
   - It is NOT a source of facts. Facts must still come only from the context.

Conversation history:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
)

# Rewrites a follow-up question into a standalone one so retrieval still works
# when the user says things like "what about his education?".
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_template(
    """
Given the conversation history and a follow-up question, rewrite the follow-up
question as a single standalone question that can be understood without the
history. Resolve pronouns and references. Do NOT answer it. If the question is
already standalone, return it unchanged.

Conversation history:
{chat_history}

Follow-up question:
{question}

Standalone question:
"""
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_sections(path: str) -> list[str]:
    """Read the knowledge base and split it into SECTION-delimited chunks."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = re.split(r"(?=SECTION: )", text)
    return [section.strip() for section in sections if section.strip()]


Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


def format_history(history: list[Message] | None) -> str:
    """Render a chat history list into a plain-text transcript for the prompt."""
    if not history:
        return "(no previous conversation)"
    lines = []
    for msg in history:
        speaker = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {msg.get('content', '')}")
    return "\n".join(lines)


def build_documents(sections: list[str]) -> list[Document]:
    """Wrap each section string in a LangChain Document with section metadata."""
    return [
        Document(
            page_content=section,
            metadata={"section": section.split("\n")[0]},
        )
        for section in sections
    ]


# --------------------------------------------------------------------------- #
# Chatbot
# --------------------------------------------------------------------------- #

class RAGChatbot:
    """Encapsulates the retriever + LLM pipeline from the notebook."""

    def __init__(
        self,
        knowledge_base_path: str = KNOWLEDGE_BASE_PATH,
        api_key: str = OPENAI_API_KEY,
    ) -> None:
        self.knowledge_base_path = knowledge_base_path
        self.api_key = api_key

        sections = load_sections(knowledge_base_path)
        self.documents = build_documents(sections)

        self.embeddings = OpenAIEmbeddings(
            api_key=api_key,
            model=EMBEDDING_MODEL,
        )

        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)

        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )

        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = RETRIEVER_K

        # Hybrid search: BM25 (keyword/sparse) + FAISS (semantic/dense)
        self.retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )

        self.llm = ChatOpenAI(
            api_key=api_key,
            model=CHAT_MODEL,
            temperature=0,
        )

        self.prompt = PROMPT
        self.contextualize_prompt = CONTEXTUALIZE_PROMPT

        # In-process conversation memory. The frontend can also pass history
        # explicitly to ask(); this is the fallback for simple CLI use.
        self.history: list[Message] = []

    def condense_question(
        self, question: str, history: list[Message] | None
    ) -> str:
        """Turn a follow-up question into a standalone one using the history."""
        if not history:
            return question
        messages = self.contextualize_prompt.invoke(
            {"chat_history": format_history(history), "question": question}
        )
        return str(self.llm.invoke(messages).content).strip() or question

    def retrieve(
        self, question: str, history: list[Message] | None = None
    ) -> list[Document]:
        """Return the top-k documents relevant to the (history-aware) question."""
        standalone = self.condense_question(question, history)
        return self.retriever.invoke(standalone)

    def ask(
        self,
        question: str,
        history: list[Message] | None = None,
        update_memory: bool = True,
    ) -> str:
        """Answer a question using retrieved context and conversation history.

        If `history` is None, the chatbot's own `self.history` is used.
        """
        if history is None:
            history = self.history

        retrieved_docs = self.retrieve(question, history)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        messages = self.prompt.invoke(
            {
                "chat_history": format_history(history),
                "context": context,
                "question": question,
            }
        )
        answer = str(self.llm.invoke(messages).content)

        if update_memory and history is self.history:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})

        return answer

    def answer_with_sources(
        self, question: str, history: list[Message] | None = None
    ) -> tuple[str, list[Document]]:
        """Like `ask`, but also return the retrieved documents (single retrieval).

        Does not touch `self.history`; the caller owns the transcript.
        """
        history = history or []
        retrieved_docs = self.retrieve(question, history)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        messages = self.prompt.invoke(
            {
                "chat_history": format_history(history),
                "context": context,
                "question": question,
            }
        )
        answer = str(self.llm.invoke(messages).content)
        return answer, retrieved_docs

    def stream_answer(
        self, question: str, history: list[Message] | None = None
    ) -> Iterator[str]:
        """Like `answer_with_sources`, but yield the answer text incrementally.

        Does not touch `self.history`; the caller owns the transcript.
        """
        history = history or []
        retrieved_docs = self.retrieve(question, history)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        messages = self.prompt.invoke(
            {
                "chat_history": format_history(history),
                "context": context,
                "question": question,
            }
        )
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield str(chunk.content)

    def reset_memory(self) -> None:
        """Clear the in-process conversation history."""
        self.history = []


if __name__ == "__main__":
    bot = RAGChatbot()
    print(bot.ask("Tell me about his Education."))
    print("---")
    print(bot.ask("Which university was that?"))  # uses memory to resolve "that"
