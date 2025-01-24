import streamlit as st
from busquedaporsimilitud import VectorSearchSystem
from settings import *


@st.cache_resource
def initialize_search_system():
    return VectorSearchSystem()

def main():
    st.title("🍹 Drink Finder Chatbot")
    
    search_system = initialize_search_system()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What kind of drink are you looking for?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching for the perfect drink..."):
                response = search_system.semantic_search(query=prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()