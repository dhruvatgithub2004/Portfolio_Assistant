"""
Streamlit frontend for the Resume Chatbot.

Run with:
    streamlit run frontend.py
"""

import streamlit as st

from backend import RAGChatbot

st.set_page_config(
    page_title="Dhruv Desai's Portfolio Assistant",
    page_icon="💼",
    layout="centered",
)


@st.cache_resource(show_spinner="Building knowledge base…")
def get_chatbot() -> RAGChatbot:
    """Build the RAG pipeline once and reuse it across reruns."""
    return RAGChatbot()


SAMPLE_QUESTIONS = [
    "Who is Dhruv Desai?",
    "What is Dhruv's current job?",
    "What certifications does Dhruv hold?",
    "Tell me about his Education.",
    "What projects has Dhruv built?",
    "How can I contact Dhruv?",
]


def main() -> None:
    st.title("💼 Dhruv Desai's Portfolio Assistant")
    st.caption(
        "Ask about Dhruv's background, skills, experience, projects, "
        "certifications, and publications. Answers are grounded only in his "
        "portfolio knowledge base."
    )

    chatbot = get_chatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Try a question")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.pending_question = q
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            chatbot.reset_memory()
            st.rerun()

    # Replay history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about Dhruv…")

    # Sample-question button click
    if not prompt and "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")

    if prompt:
        # Conversation history = everything said before this new question.
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer = st.write_stream(chatbot.stream_answer(prompt, history))

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
