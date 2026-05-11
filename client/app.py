import streamlit as st
import requests

st.title("Drive Chat Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    res = requests.post(
        "http://localhost:8000/chat",
        json={"messages": st.session_state.messages}
    )
    data = res.json()

    bot_reply = data["response"]
    files = data.get("files", [])

    st.chat_message("assistant").write(bot_reply)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    if files:
        for f in files:
            st.markdown(f"- [{f['name']}]({f.get('webViewLink', '#')})")
