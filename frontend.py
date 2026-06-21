# frontend.py
import streamlit as st
import backend
from backend import process_user_query


def render_ui():
    """
    Renders the Gemini-style user interface.
    """
    # Main content at the top
    st.title("🤖 Pauna medical assistant")
    st.write("Hi, my name is Pauna! I'm a pulmonary specialist at ACTREC. I'm here to help you with your medical questions.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]

    if len(st.session_state.chat_history) > 0:
        st.write(backend.make_prompt())

    # Empty container to hold the response output below the title
    response_placeholder = st.empty()
    
    # Pinned Bottom Input Area
    with st.container():
        st.caption("type your query here")
        user_query = st.chat_input("Enter your query here (Shift+Enter for new line)...")
        
    # Trigger processing when the user submits a query
    if user_query:
        st.session_state.chat_history.append({'user':'[INST]' + user_query + '[/INST]'})
        # 1. Send the input to backend.py to be printed/processed
        backend_response = process_user_query(user_query)
        context = next(backend_response)

        # 2. Display the result in the upper content area
        with response_placeholder.container():
            st.info(f"**Your Query:**\n\n{user_query} \n\n context:\n\n {context}")
            op=st.write_stream(backend_response)
            st.session_state.chat_history.append({"model": op})