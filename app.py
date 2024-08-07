import streamlit as st
from bot import chatbot

st.markdown("<h3 style='text-align: center;'>Curhat Dong BOT</h3><h6 style='text-align: center;'>Iyaaa MBOT</h6>", unsafe_allow_html=True)

# Inisialisasi session_state untuk menyimpan pesan-pesan chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "👩", "content": "Kamu dateng kesini? ada masalah kah, ceritain aja?!"})

# Tampilkan pesan-pesan yang sudah ada di session_state
for message in st.session_state.messages:
    if message["role"] == "😣":
        with st.chat_message(message["role"]):
            st.warning(message["content"])
    elif message["role"] == "👩":
        with st.chat_message(message["role"]):
            st.success(message["content"])

prompt = st.chat_input("Ask Him...")
if prompt:
    bot_response = chatbot(prompt)

    with st.chat_message("😣"):
        st.warning(prompt)

    with st.chat_message("👩"):
        st.success(bot_response)

    # Simpan pesan pengguna dan respons bot ke dalam session_state
    st.session_state.messages.append({"role": "😣", "content": prompt})
    st.session_state.messages.append({"role": "👩", "content": bot_response})
