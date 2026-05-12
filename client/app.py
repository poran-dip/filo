import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

    def get_icon(mime):
        return (
            "📄" if "document" in mime else
            "📊" if "spreadsheet" in mime else
            "📑" if "pdf" in mime else
            "🖼️" if "image" in mime else
            "📁" if "folder" in mime else
            "📎"
        )

    def format_size(size):
        if not size:
            return "—"
        size = int(size)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 ** 2:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / 1024 ** 2:.1f} MB"

    rows = ""
    for f in files:
        icon = get_icon(f.get("mimeType", ""))
        name = f.get("name", "Untitled")
        link = f.get("webViewLink", "#")
        modified = f.get("modifiedTime", "")[:10]
        size = format_size(f.get("size"))
        rows += f"""
        <tr>
            <td style="font-size:1.2rem; text-align:center">{icon}</td>
            <td><a href="{link}" target="_blank" style="text-decoration:none">{name}</a></td>
            <td style="color:#888; font-size:0.85rem">{modified}</td>
            <td style="color:#888; font-size:0.85rem">{size}</td>
        </tr>
        """

    table_html = f"""
    <style>
    * {{ font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }}
    </style>
    <table style="width:100%; border-collapse:collapse; margin-top:8px">
        <thead>
            <tr style="border-bottom:1px solid #e0e0e0; color:#555; font-size:0.8rem; text-transform:uppercase">
                <th style="width:40px"></th>
                <th style="text-align:left; padding:6px 0">Name</th>
                <th style="text-align:left; padding:6px 0">Modified</th>
                <th style="text-align:left; padding:6px 0">Size</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """
    st.iframe(table_html, height=min(80 + len(files) * 48, 600))

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
