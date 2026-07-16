import streamlit as st
import os
import requests

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000") + "/chat"

st.title("Multi-agent Assistant")
st.caption("Powered by math, weather, news, recipes, jokes, translation, dictionary, and currency agents")

# Session identity
with st.sidebar:
    st.header("Session")
    username = st.text_input("Your username", value="guest")
    thread_id = f"streamlit-{username}"
    st.caption(f"Thread: `{thread_id}`")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            split_result = message["content"].split(" - ")
            st.caption(split_result[0])
            st.markdown(split_result[1])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={
                    "message": prompt,
                    "thread_id": thread_id
                })
                result = response.json()["response"]
            except Exception as e:
                result = f"Error: {e}"
        split_result = result.split(" - ")
        st.caption(split_result[0])
        st.markdown(split_result[1])
        st.session_state.messages.append({"role": "assistant", "content": result})
