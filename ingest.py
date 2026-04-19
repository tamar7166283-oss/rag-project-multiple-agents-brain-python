import os
import ssl
import urllib3
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. Load environment variables from .env file
load_dotenv()

# --- SSL & NETWORK FIX (Required for filtered networks like NetFree) ---
# This tells the system not to strictly check the local issuer certificate
os.environ['CURL_CA_BUNDLE'] = '' 
# Disables the insecure request warnings in the terminal
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# -----------------------------------------------------------------------

def run_indexing_pipeline():
    # Setup project paths
    project_path = os.getenv("TARGET_PROJECT_PATH", "../DEMO-PROJECT")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.abspath(os.path.join(base_dir, project_path))

    # --- Step 1: Load Documents ---
    print(f"📂 Loading documents from: {full_path}")
    reader = SimpleDirectoryReader(
        input_dir=full_path, 
        recursive=True, 
        required_exts=[".md", ".mdc"]
    )
    documents = reader.load_data()

    # --- Step 2: Transform & Clean (The Pipeline) ---
    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(), 
            SentenceSplitter(chunk_size=512, chunk_overlap=50)
        ]
    )
    
    print("⚙️  Parsing and splitting documents...")
    raw_nodes = pipeline.run(documents=documents)

    # Filtering nodes that are too short to ensure quality context
    MIN_NODE_LENGTH = 50 
    valid_nodes = [
        node for node in raw_nodes 
        if len(node.get_content().strip()) > MIN_NODE_LENGTH
    ]
    print(f"🧹 Filtering complete: {len(valid_nodes)} high-quality nodes remain.")

    # --- Step 3: Setup Embedding Model (Cohere) ---
    print("🧬 Initializing Cohere Multilingual Embedding...")
    embed_model = CohereEmbedding(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model_name="embed-multilingual-v3.0",
        input_type="search_document"
    )

    # --- Step 4: Setup Vector Store (Pinecone) ---
    print("🌲 Connecting to Pinecone...")
    
    # Initialize Pinecone with ssl_verify=False to bypass connection errors
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        ssl_verify=False  # <--- CRITICAL FIX for SSL/Network issues
    )
    
    index_name = "agent-docs-index"
    pinecone_index = pc.Index(index_name)
    
    # Link Pinecone to LlamaIndex
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Step 5: Indexing (Sending data to the Cloud) ---
    print(f"📤 Uploading {len(valid_nodes)} nodes to Pinecone. Please wait...")
    
    # This process embeds the text and saves it to the vector database
    index = VectorStoreIndex(
        nodes=valid_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("\n🚀 SUCCESS! Your knowledge base is indexed and ready in the cloud.")
    return index

if __name__ == "__main__":
    try:
        run_indexing_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")