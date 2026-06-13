from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os

def create_vector_db(pdf_path: str):
    # Load the PDF document
    persist_directory = r"/home/kaushal-chari/capstone/Medical_Assistant/medical_dir"
    doc_loader = PyMuPDFLoader(r"/home/kaushal-chari/capstone/Medical_Assistant/medical_diagnosis_manual.pdf")
    doc = doc_loader.load()
    test_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,
                                            chunk_overlap = 200)
    chunks = doc_loader.load_and_split(text_splitter=test_splitter)
    embedding_model = HuggingFaceEmbeddings(model_name = r"sentence-transformers/all-MiniLM-L6-V2")
    persist_storage = "/home/kaushal-chari/capstone/Medical_Assistant/medical_dir"
    batch_size = 500
    vector_db = Chroma.from_documents(documents=chunks[0:batch_size],
        embedding=embedding_model,
        persist_directory= persist_storage,
        collection_name="medical_vector_store")
    print("completed 1 batch")
    for i in range(batch_size, len(chunks), batch_size):
        print(f"batch number {i}")
        vector_db.add_documents(chunks[i:i + batch_size])

if __name__=="__main__":
    create_vector_db("/home/kaushal-chari/capstone/Medical_Assistant/medical_diagnosis_manual.pdf")        