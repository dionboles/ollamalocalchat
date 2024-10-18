import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Bot", page_icon="🤖")
st.title("Streaming Bot")


def get_response(query, chat_history):
    template = [
        (
            "system",
            """You are a helpful assistant. Answer the following
question considering the history of the conversation: Chat history
{chat_history}""",
        ),
        ("human", "{user_question}"),
    ]

    prompt = ChatPromptTemplate(template)

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"chat_history": chat_history, "user_question": query})
    return response


# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Your message")
if user_query is not None and user_query != "":
    new_human_message = HumanMessage(user_query)
    st.session_state.chat_history.append(new_human_message)

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
    # Get AI response for previous human message
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)
    # Append new AI message to chat history
    st.session_state.chat_history.append(AIMessage(ai_response))
