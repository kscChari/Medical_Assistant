from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

embedding_model = HuggingFaceEmbeddings(model_name= r'sentence-transformers/all-MiniLM-L6-V2')
main_db = chromadb.PersistentClient(path = r"/home/kaushal-chari/capstone/Medical_Assistant/medical_dir")
print(main_db.list_collections())
collection=main_db.get_collection("medical_vector_store")
docs=collection.get(offset=1050,limit=1, include=["documents"])
print(docs["documents"])
