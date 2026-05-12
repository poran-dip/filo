import requests
import streamlit as st
from client.config import BACKEND_URL


st.set_page_config(page_title="Drive Chat Agent", page_icon="📁")
st.title("📁 Drive Chat Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_results" not in st.session_state:
    st.session_state.file_results = {}

def _render_files(files: list[dict]):
    if not files:
        return
    st.markdown("**Files found:**")
    if len(files) <= 6:
        cols_per_row = 2
        for i in range(0, len(files), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, file in enumerate(files[i : i + cols_per_row]):
                with cols[j]:
                    mime = file.get("mimeType", "")
                    icon = (
                        "📄" if "document" in mime else
                        "📊" if "spreadsheet" in mime else
                        "📑" if "pdf" in mime else
                        "🖼️" if "image" in mime else
                        "📁"
                    )
                    modified = file.get("modifiedTime", "")[:10]
                    link = file.get("webViewLink", "#")
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e0e0e0; border-radius:8px; padding:12px; margin:4px 0">
                            <div style="font-size:1.5rem">{icon}</div>
                            <div style="font-weight:600; margin:4px 0">{file['name']}</div>
                            <div style="color:#888; font-size:0.8rem">Modified: {modified}</div>
                            <a href="{link}" target="_blank">🔗 Open in Drive</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        for f in files:
            mime = f.get("mimeType", "")
            icon = (
                "📄" if "document" in mime else
                "📊" if "spreadsheet" in mime else
                "📑" if "pdf" in mime else
                "🖼️" if "image" in mime else
                "📁"
            )
            st.markdown(f"{icon} [{f['name']}]({f.get('webViewLink', '#')}) — {f.get('modifiedTime', '')[:10]}")

# Render history
for i, msg in enumerate(st.session_state.messages):
    st.chat_message(msg["role"]).write(msg["content"])
    if msg["role"] == "assistant" and i in st.session_state.file_results:
        _render_files(st.session_state.file_results[i])

user_input = st.chat_input("Ask about your Drive files...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Searching..."):
        try:
            res = requests.post(
                f"{BACKEND_URL}/chat",
                json={"messages": st.session_state.messages},
            )
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.session_state.messages.pop()
            st.stop()

    bot_reply = data["response"]
    files = data.get("files", [])

    st.chat_message("assistant").write(bot_reply)
    turn_index = len(st.session_state.messages)  # index BEFORE appending assistant
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    if files:
        st.session_state.file_results[turn_index] = files
        _render_files(files)
