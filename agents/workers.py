from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

def extract_tokens(message) -> dict:
    usage = message.response_metadata.get("token_usage", {}) or message.response_metadata.get("usage", {})
    return {
        "prompt": usage.get("prompt_tokens", 0),
        "completion": usage.get("completion_tokens", 0),
        "total": usage.get("total_tokens", 0)
    }

def create_retriever_node(llm, system_prompt: str, tools):
    """Runs a ReAct loop internally, captures logs/tokens, and returns a clean summary."""
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    
    def node(state):
        result = agent.invoke({"messages": state["messages"]})
        
        tool_logs = []
        prompt_tokens, comp_tokens, total_tokens = 0, 0, 0
        final_text = "No additional findings retrieved."
        
        new_messages = result["messages"][len(state["messages"]):]
        for m in new_messages:
            if isinstance(m, AIMessage):
                tokens = extract_tokens(m)
                prompt_tokens += tokens["prompt"]
                comp_tokens += tokens["completion"]
                total_tokens += tokens["total"]
                
                if m.tool_calls:
                    for tc in m.tool_calls:
                        tool_logs.append({
                            "agent": "Contextual Retriever",
                            "type": "call",
                            "tool": tc["name"],
                            "query": str(tc.get("args", {}))
                        })
                if m.content:
                    final_text = m.content
                    
            elif isinstance(m, ToolMessage):
                tool_logs.append({
                    "agent": "Contextual Retriever",
                    "type": "result",
                    "tool": m.name,
                    "result": m.content
                })
                
        # Return cleanly constructed AIMessage to prevent 400 schema error
        clean_msg = AIMessage(content=final_text, name="contextual_retriever")
        calls_count = len([l for l in tool_logs if l['type'] == 'call'])
        
        return {
            "messages": [clean_msg],
            "tool_logs": tool_logs,
            "token_usage": {"contextual_retriever": {"prompt": prompt_tokens, "completion": comp_tokens, "total": total_tokens}},
            "task_summaries": [f"**Contextual Retriever**: Executed {calls_count} search tool actions and synthesized raw evidence."]
        }
    return node

def create_agent_node(llm, system_prompt: str, agent_name: str):
    """Standard execution node for Analyzer, Insight, and Report Builder."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm
    
    def node(state):
        result = chain.invoke(state)
        tokens = extract_tokens(result)
        
        clean_msg = AIMessage(content=result.content, name=agent_name)
        return {
            "messages": [clean_msg],
            "token_usage": {agent_name: tokens},
            "task_summaries": [f"**{agent_name.replace('_', ' ').title()}**: Completed its analytical reasoning step."]
        }
    return node