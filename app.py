import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ui.components import render_sidebar, render_chat_history
from tools.retrievers import setup_local_index
from agents.graph import build_workflow_graph

st.set_page_config(page_title="Multi-Agent Deep Researcher", page_icon="🧠", layout="wide")
st.title("🧠 Multi-Agent Deep Researcher")

if "messages" not in st.session_state:
    st.session_state.messages = []

openrouter_key, tavily_key, model_id, uploaded_files, process_btn, clear_btn = render_sidebar()

if process_btn and uploaded_files:
    with st.spinner("Indexing PDFs via LlamaIndex & FAISS..."):
        temp_dir = tempfile.mkdtemp()
        for f in uploaded_files:
            with open(os.path.join(temp_dir, f.name), "wb") as disk_file:
                disk_file.write(f.read())
        try:
            setup_local_index(temp_dir)
            st.sidebar.success("✅ PDFs indexed successfully!")
        except Exception as e:
            st.sidebar.error(f"Indexing error: {e}")

if clear_btn:
    st.session_state.messages = []
    st.rerun()

if tavily_key:
    os.environ["TAVILY_API_KEY"] = tavily_key

render_chat_history(st.session_state.messages)

if prompt := st.chat_input("Ask a research question..."):
    if not openrouter_key:
        st.error("Please enter your OpenRouter API Key.")
        st.stop()

    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(
        api_key=openrouter_key, 
        base_url="https://openrouter.ai/api/v1",
        model_name=model_id, 
        temperature=0.1,
    )
    graph = build_workflow_graph(llm)

    with st.chat_message("assistant"):
        expander = st.expander("🔍 Real-time Agent Execution & Tool Routing", expanded=True)
        final_placeholder = st.empty()
        summary_placeholder = st.empty()
        
        final_token_usage = {}
        all_task_summaries = []
        
        try:
            events = graph.stream(
                {"messages": st.session_state.messages, "tool_logs": [], "token_usage": {}, "task_summaries": []},
                config={"recursion_limit": 30}
            )
            
            for event in events:
                for node_name, state_update in event.items():
                    
                    # Track Tokens
                    if "token_usage" in state_update:
                        for agent, usage in state_update["token_usage"].items():
                            if agent not in final_token_usage:
                                final_token_usage[agent] = {"prompt": 0, "completion": 0, "total": 0}
                            final_token_usage[agent]["prompt"] += usage["prompt"]
                            final_token_usage[agent]["completion"] += usage["completion"]
                            final_token_usage[agent]["total"] += usage["total"]
                    
                    # Track Task Summaries
                    if "task_summaries" in state_update:
                        all_task_summaries.extend(state_update["task_summaries"])
                    
                    # Render Supervisor routing & Tool executions
                    if node_name == "supervisor":
                        expander.markdown(f"🧭 **Supervisor:** Route to `{state_update.get('next', 'UNKNOWN')}`")
                        
                    if "tool_logs" in state_update:
                        for log in state_update["tool_logs"]:
                            if log["type"] == "call":
                                expander.markdown(f"⚙️ **{log['agent']}** queried `{log['tool']}`: *\"{log['query']}\"*")
                            elif log["type"] == "result":
                                expander.markdown(f"📄 **Result [{log['tool']}]:** `{str(log['result'])[:200]}...`")
                    
                    # Render final agent steps and outputs
                    if "messages" in state_update and len(state_update["messages"]) > 0:
                        last_msg = state_update["messages"][-1]
                        if last_msg.name:
                            expander.markdown(f"🤖 **{last_msg.name}** finished processing.")
                        if node_name == "report_builder":
                            final_placeholder.markdown(last_msg.content)
                            st.session_state.messages.append(last_msg)

            # Render Metrics & Bullet Summaries at the end of execution
            total_tokens = sum([v["total"] for v in final_token_usage.values()])
            
            summary_md = "### 📋 Execution Summary & Token Usage\n\n**Task Highlights:**\n"
            for task in all_task_summaries:
                summary_md += f"- {task}\n"
                
            summary_md += "\n**Token Consumption:**\n"
            for agent, usage in final_token_usage.items():
                summary_md += f"- **{agent.replace('_', ' ').title()}**: {usage['total']:,} tokens *(Prompt: {usage['prompt']:,}, Completion: {usage['completion']:,})*\n"
            summary_md += f"\n**Total Session Tokens: {total_tokens:,}**"
            
            summary_placeholder.info(summary_md)

        except Exception as e:
            st.error(f"Execution error: {e}")