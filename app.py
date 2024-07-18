import streamlit as st
from bot import chatbot


st.markdown("<h1 style='text-align: center; color: gold;'>LORD OF THE RINGS</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "💪", "content": "Welcome champ, I am the Lord of the Rings, ready to assist you with gymnastic rings workouts, exercises, and nutrition advice. Feel free to ask me!"})

for message in st.session_state.messages:
    if message["role"] == "👤":
        with st.chat_message(message["role"]):
            st.success(message["content"])
    elif message["role"] == "💪":
        with st.chat_message(message["role"]):
            st.warning(message["content"])

prompt = st.chat_input("Ask Him...")
if prompt:
    bot_response = chatbot(prompt)

    with st.chat_message("👤"):
        st.success(prompt)

    with st.chat_message("💪"):
        st.warning(bot_response)

    st.session_state.messages.append({"role": "👤", "content": prompt})
    st.session_state.messages.append({"role": "💪", "content": bot_response})