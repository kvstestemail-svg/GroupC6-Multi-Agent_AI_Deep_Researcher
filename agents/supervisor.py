from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal

class Route(BaseModel):
    reasoning: str
    next: Literal["contextual_retriever", "critical_analysis", "insight_generation", "report_builder", "FINISH"]

def get_supervisor_node(llm):
    members = ["contextual_retriever", "critical_analysis", "insight_generation", "report_builder"]
    options = ["FINISH"] + members

    system_prompt = (
        "You are the Research Supervisor managing a collaborative workflow: {members}.\n"
        "STRICT ROUTING RULES:\n"
        "1. Start with 'contextual_retriever' to gather data using search tools.\n"
        "2. If retrieval is complete, route to 'critical_analysis' to evaluate credibility.\n"
        "3. After critical analysis, route to 'insight_generation' for trend synthesis.\n"
        "4. After insight generation, route to 'report_builder' to synthesize all findings.\n"
        "5. Choose 'FINISH' ONLY AFTER 'report_builder' has delivered the final formatted report.\n"
        "DO NOT SKIP STEPS. NEVER select 'FINISH' immediately after 'contextual_retriever'."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Who should act next? Select exactly one of: {options}")
    ]).partial(options=str(options), members=", ".join(members))

    # include_raw=True guarantees we can extract token metadata from the supervisor
    chain = prompt | llm.with_structured_output(Route, include_raw=True)

    def supervisor_node(state):
        res = chain.invoke(state)
        parsed = res["parsed"]
        raw = res["raw"]
        
        usage = raw.response_metadata.get("token_usage", {}) or raw.response_metadata.get("usage", {})
        
        return {
            "next": parsed.next,
            "token_usage": {"supervisor": {
                "prompt": usage.get("prompt_tokens", 0), 
                "completion": usage.get("completion_tokens", 0), 
                "total": usage.get("total_tokens", 0)
            }},
            "task_summaries": [f"**Supervisor**: Evaluated state and routed execution to `{parsed.next}`."]
        }
    return supervisor_node