import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

def merge_token_usage(dict1: dict, dict2: dict) -> dict:
    """Safely merges and accumulates token counts across multiple agent runs."""
    d = dict1.copy()
    for agent, usage in dict2.items():
        if agent in d:
            d[agent] = {
                "prompt": d[agent].get("prompt", 0) + usage.get("prompt", 0),
                "completion": d[agent].get("completion", 0) + usage.get("completion", 0),
                "total": d[agent].get("total", 0) + usage.get("total", 0)
            }
        else:
            d[agent] = usage
    return d

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    tool_logs: Annotated[list, operator.add]
    token_usage: Annotated[dict, merge_token_usage]
    task_summaries: Annotated[list, operator.add]