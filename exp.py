import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import os

db_path = r"/home/kaushal-chari/capstone/Medical_Assistant/medical_dir"

if os.path.exists(db_path):
    print(f"Directory exists! Contents: {os.listdir(db_path)}")
else:
    print(f"ERROR: Path does not exist: {db_path}")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")

db = Chroma(
    collection_name='medical_vector_store',
    embedding_function=embedding_model, 
    persist_directory="/home/kaushal-chari/capstone/Medical_Assistant/medical_dir"
)

# 1. Check document count
try:
    count = db._collection.count()
    print(f"Total documents found in Chroma: {count}")
except Exception as e:
    print(f"Error reading collection count: {e}")

# 2. Peek into the database
try:
    data_peek = db._collection.peek(limit=1)
    print("\n--- Database Peek Result ---")
    print(data_peek)
except Exception as e:
    print(f"Error peeking into collection: {e}")

context = [7]
if context:
    print("context exists")