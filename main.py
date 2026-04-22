import os
import urllib3
import gradio as gr
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# ייבוא מה-Config
from config import embed_model, llm, PINECONE_API_KEY, PINECONE_INDEX_NAME


os.environ['CURL_CA_BUNDLE'] = '' 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

Settings.llm = llm
Settings.embed_model = embed_model

def setup_rag():
    print("🌲 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY, ssl_verify=False)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )

    retriever = VectorIndexRetriever(
        index=index, 
        similarity_top_k=5
    )
    
    response_synthesizer = get_response_synthesizer(
        llm=llm, 
        response_mode="compact"
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)] # השארתי נמוך לבדיקה
    )

try:
    engine = setup_rag()
    print("✅ RAG Engine is ready!")
except Exception as e:
    print(f"❌ Failed to initialize RAG: {e}")
    engine = None

def chat(message, history):
    if engine is None:
        return "שגיאה בחיבור למסד הנתונים."
    try:
        print(f"🔍 Searching for: {message}")
        response = engine.query(message)
        print(f"📚 Found {len(response.source_nodes)} relevant nodes.")
        return str(response)
    except Exception as e:
        return f"קרתה שגיאה: {e}"

gr.ChatInterface(fn=chat, title="Agentic RAG Explorer").launch()