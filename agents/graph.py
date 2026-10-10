from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.supervisor import get_supervisor_node
from agents.workers import create_retriever_node, create_agent_node
from tools.retrievers import get_all_tools

def build_workflow_graph(llm):
    workflow = StateGraph(AgentState)
    tools = get_all_tools()

    workflow.add_node("supervisor", get_supervisor_node(llm))

    retriever_node = create_retriever_node(
        llm, 
        system_prompt=(
            "You are the Contextual Retriever. Use `local_report_search`, `arxiv_search`, `tavily_search`, and `wikipedia_search`.\n"
            "Query multiple tools concurrently. Output a highly detailed compilation of raw evidence, data points, and URLs."
        ), 
        tools=tools
    )

    analysis_node = create_agent_node(
        llm, 
        system_prompt=(
            "You are the Critical Analysis Agent. Review the data gathered by the Contextual Retriever.\n"
            "- Identify contradictions between sources.\n"
            "- Flag methodological weaknesses or source bias.\n"
            "- Output analytical bullet points."
        ),
        agent_name="critical_analysis"
    )

    insight_node = create_agent_node(
        llm, 
        system_prompt=(
            "You are the Insight Generation Agent. Review the findings and the critical analysis.\n"
            "- Formulate predictive hypotheses using reasoning chains.\n"
            "- Highlight macro-trends and strategic implications."
        ),
        agent_name="insight_generation"
    )

    report_node = create_agent_node(
        llm, 
        system_prompt=(
            "You are the Report Builder. Synthesize all retrieved data, critique, and insights into a publication-ready markdown report.\n"
            "- Include an Executive Summary, Key Findings, Critical Tensions, Strategic Insights, and Conclusion.\n"
            "- Embed clickable source URLs and citations."
        ),
        agent_name="report_builder"
    )

    workflow.add_node("contextual_retriever", retriever_node)
    workflow.add_node("critical_analysis", analysis_node)
    workflow.add_node("insight_generation", insight_node)
    workflow.add_node("report_builder", report_node)

    for member in ["contextual_retriever", "critical_analysis", "insight_generation", "report_builder"]:
        workflow.add_edge(member, "supervisor")

    workflow.add_conditional_edges("supervisor", lambda x: x["next"], {
        "contextual_retriever": "contextual_retriever",
        "critical_analysis": "critical_analysis",
        "insight_generation": "insight_generation",
        "report_builder": "report_builder",
        "FINISH": END
    })

    workflow.add_edge(START, "supervisor")
    return workflow.compile()