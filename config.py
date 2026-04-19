import os
from dotenv import load_dotenv
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere

# טעינת משתני סביבה מה-env
load_dotenv()
os.environ['CURL_CA_BUNDLE'] = '' # תיקון נטפרי

# הגדרות Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "agent-docs-index" 

# נתיב לפרויקט היעד - ודאי שהנתיב הזה קיים במחשב שלך!
PROJECT_PATH = os.getenv("TARGET_PROJECT_PATH", "../DEMO-PROJECT")

# הגדרת המודלים
embed_model = CohereEmbedding(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-multilingual-v3.0"
)

llm = Cohere(
    api_key=os.getenv("COHERE_API_KEY"), 
    model="command-r-plus-08-2024" 
)

print(f"✅ Config.py נטען בהצלחה")