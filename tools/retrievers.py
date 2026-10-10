import os
import faiss
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

local_retriever = None

def setup_local_index(pdf_dir: str):
    """Initializes LlamaIndex and FAISS with uploaded PDFs."""
    global local_retriever
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    docs = SimpleDirectoryReader(pdf_dir).load_data()
    faiss_index = faiss.IndexFlatL2(384) # Dimension for bge-small-en-v1.5
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    local_retriever = index.as_retriever(similarity_top_k=3)

@tool
def local_report_search(query: str) -> str:
    """Searches through the user's uploaded local PDF reports."""
    global local_retriever
    if local_retriever is None:
        return "System Message: No local documents have been indexed."
    results = local_retriever.retrieve(query)
    if not results:
        return "No relevant passages were found in the indexed reports."
    return "\n\n".join(
        f"Source: {result.metadata.get('file_name', result.metadata.get('file_path', 'Unknown source'))}\n"
        f"{result.get_content()}"
        for result in results
    )

@tool
def tavily_search(query: str) -> str:
    """Searches the internet for news, current developments, and real-time data."""
    if not os.environ.get("TAVILY_API_KEY"):
        return "System Message: Tavily API key is missing."
    try:
        search_tool = TavilySearchResults(max_results=3)
        return str(search_tool.invoke({"query": query}))
    except Exception as e:
        return f"Error executing internet search: {str(e)}"

wikipedia_search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv_search = ArxivQueryRun()

def get_all_tools():
    return [local_report_search, tavily_search, wikipedia_search, arxiv_search]