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

# --- תיקוני רשת לנטפרי ---
os.environ['CURL_CA_BUNDLE'] = '' 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# הגדרת מודלים גלובלית
Settings.llm = llm
Settings.embed_model = embed_model

def setup_rag():
    print("🌲 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY, ssl_verify=False)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # טעינת האינדקס מה-Vector Store
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        embed_model=embed_model
    )

    # --- כאן הייתה הבעיה! הגדרת ה-Retriever ---
    retriever = VectorIndexRetriever(
        index=index, 
        similarity_top_k=5
    )
    
    # הגדרת המנסח (Synthesizer)
    response_synthesizer = get_response_synthesizer(
        llm=llm, 
        response_mode="compact"
    )

    # בניית מנוע השאילתות - שימי לב שכל המשתנים מוגדרים עכשיו
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.1)] # השארתי נמוך לבדיקה
    )

# אתחול המנוע
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

# הפעלה
gr.ChatInterface(fn=chat, title="Agentic RAG Explorer").launch()