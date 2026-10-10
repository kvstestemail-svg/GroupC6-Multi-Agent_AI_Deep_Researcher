import streamlit as st

def render_sidebar():
    st.sidebar.title("⚙️ Researcher Setup")
    
    openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")
    tavily_key = st.sidebar.text_input("Tavily API Key (Optional)", type="password")
    model_id = st.sidebar.text_input("OpenRouter Model ID", value="openai/gpt-4o")
    
    st.sidebar.divider()
    uploaded_files = st.sidebar.file_uploader("Upload Reports for Analysis", type=["pdf"], accept_multiple_files=True)
    process_btn = st.sidebar.button("Index Documents")
    clear_btn = st.sidebar.button("Clear Conversation")
    
    return openrouter_key, tavily_key, model_id, uploaded_files, process_btn, clear_btn

def render_chat_history(messages):
    for msg in messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai" and msg.name == "report_builder" and msg.content:
            st.chat_message("assistant").write(msg.content)