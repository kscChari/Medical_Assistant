# backend.py
from llama_cpp import Llama
from typing import Generator
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

def make_prompt():
    final_prompt = ''
    for i in st.session_state.chat_history:
        if 'user' in i.keys():
            final_prompt = final_prompt + "\n\n"+'[INST]'+"user\n\n:"+i['user'] + '[/INST]'+  '\n'
        if 'model' in i.keys():
            final_prompt = final_prompt + '\n\n' + 'model:\n\n' +i['model'] + '\n'
        if 'context' in i.keys():
            final_prompt = final_prompt + '[INST]'+ '\n\n'+"context:" + "\n\n" + i['context']  +'[/INST]'+ "\n\n"
    return final_prompt
            

@st.cache_resource
def create_llm_object():
    llm = Llama(model_path = "model/mistral-7b-instruct-v0.2.Q6_K.gguf",
            n_ctx = 4096,
            n_threads = 8,
            n_gpu_layers = 10)
    return llm
llm = create_llm_object()
system_prompt = """[INST]You are a helpful and precise assistant for answering medical questions.
Your name is Pauna, always introduce yourself as DR. Pauna before answering the query. You will be given a medical question, and you will provide a detailed and accurate answer based on your expertise in medicine. Always ensure that your responses are clear, concise, and informative.
Appriciate the user's bravery for asking the question and provide a detailed answer to the best of your ability. If you don't know the answer, say you don't know but suggest possible next steps for the user to find the information they need.[/INST]"""
embedding_model = HuggingFaceEmbeddings(model_name=r"/home/kaushal-chari/capstone/Medical_Assistant/L6-V2-Model")
db = Chroma(collection_name= 'medical_vector_store',embedding_function=embedding_model, persist_directory=r"/home/kaushal-chari/capstone/Medical_Assistant/medical_dir")
retriever = db.as_retriever()
def process_user_query(query: str) -> Generator[str, None, None]:
    retriever.search_type="similarity_score_threshold"
    retriever.search_kwargs = {'score_threshold': 0.5}
    context =[]
    if len(st.session_state.chat_history) == 1:
        context = retriever.invoke(query)
        st.session_state.chat_history.append({'context': context[0].page_content})
        print("the retrieved constext is given here:\n" + context[0].page_content)
    #print(f'context length is {len(context)}')
        doc_scores = db.similarity_search_with_relevance_scores(query=query, k =1)
        score = doc_scores[0][1]
        print(doc_scores[0][0].page_content)
        print(f'the score was: {score}')

    if context:
        yield context[0].page_content
        prompt =  system_prompt + "\n\n"  + make_prompt() + '\n'
        print("="*30)
        print("PROMPT IS: \n")
        print(prompt)
        print("="*30)     
        output = llm(prompt=prompt,
                 max_tokens=1000,
                 temperature=0.0,
                 stream=True)
        print(context[0].page_content)
    else:
        yield "no relevant data found"
        output = llm(prompt=system_prompt + "\n\n" + make_prompt(),
                 max_tokens=1000,
                 temperature=0.0,
                 stream=True)
    # This prints directly to your terminal screen where Streamlit is running
    print(f"\n[BACKEND LOG] Received user query:\n{query}\n")


    for token in output:
    # Return a message back to the frontend
        chunk = token['choices'][0]['text']
        yield chunk
