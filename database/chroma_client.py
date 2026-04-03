import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

def get_client():
    """ChromaDB için persistent bir client döndürür."""
    # Verilerin kaydedileceği dizini kontrol et
    persist_dir = "./data/chroma"
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)
        
    client = chromadb.PersistentClient(path=persist_dir)
    return client

def get_collection(name="code_chunks"):
    """Koleksiyonu döndürür, yoksa oluşturur."""
    client = get_client()
    # get_or_create_collection sayesinde fonksiyon her çağrıldığında güvenle çalışır
    collection = client.get_or_create_collection(name=name)
    return collection