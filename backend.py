# backend.py
from llama_cpp import Llama
from typing import Generator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

llm = Llama(model_path = "model/mistral-7b-instruct-v0.2.Q6_K.gguf",
            n_ctx = 4096,
            n_threads = 8,
            n_gpu_layers = 10)
system_prompt = """[INST]You are a helpful and precise assistant for answering medical questions.
Your name is Pauna, always introduce yourself as DR. Pauna before answering the query. You will be given a medical question, and you will provide a detailed and accurate answer based on your expertise in medicine. Always ensure that your responses are clear, concise, and informative.
Appriciate the user's bravery for asking the question and provide a detailed answer to the best of your ability. If you don't know the answer, say you don't know but suggest possible next steps for the user to find the information they need."""
embedding_model = HuggingFaceEmbeddings(model_name=r"/home/kaushal-chari/capstone/Medical_Assistant/L6-V2-Model")
db = Chroma(collection_name= 'medical_vector_store',embedding_function=embedding_model, persist_directory=r"/home/kaushal-chari/capstone/Medical_Assistant/medical_dir")
retriever = db.as_retriever()
def process_user_query(query: str) -> Generator[str, None, None]:
    retriever.search_type="similarity_score_threshold"
    context = retriever.invoke(query, config={'search_kwargs':{'score_threshold': 0.7}})
    print(len(context))

    if context:
        yield context[0].page_content        
        output = llm(prompt=system_prompt + "\n\n" + "context:\n"+ context[0].page_content + query + " [/INST]",
                 max_tokens=1000,
                 temperature=0.0,
                 stream=True)
        print(context[0].page_content)
    else:
        yield "no relevant data found"
        output = llm(prompt=system_prompt + "\n\n" + query + " [/INST]",
                 max_tokens=1000,
                 temperature=0.0,
                 stream=True)
    # This prints directly to your terminal screen where Streamlit is running
    print(f"\n[BACKEND LOG] Received user query:\n{query}\n")


    for token in output:
    # Return a message back to the frontend
        chunk = token['choices'][0]['text']
        yield chunk
