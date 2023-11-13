import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers, Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! This is a contract evaluation tool"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_embeddings(text_chunks):
    # Ensure text_chunks are in the correct format
    if not all(isinstance(chunk, str) for chunk in text_chunks):
        raise ValueError("All elements in text_chunks must be strings.")

    # Initialize the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': 'cpu'})

    # Ensure no null or malformed text chunks
    cleaned_text_chunks = [chunk.replace("\n", " ") if chunk else "" for chunk in text_chunks]

    return embeddings.embed_documents(cleaned_text_chunks)

    loader = None
    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension in [".docx", ".doc"]:
        loader = Docx2txtLoader(temp_file_path)
    elif file_extension == ".txt":
        loader = TextLoader(temp_file_path)

        if loader:
        text.extend(loader.load())
        os.remove(temp_file_path)
    return text

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_embeddings(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': 'cpu'})
    return embeddings.embed_documents(text_chunks)

def main():
    load_dotenv()
    initialize_session_state()
    st.title("Evaluation tool for contracts sustainability evaluation ")
    st.sidebar.title("Drop your contract documents here ")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = process_uploaded_files(uploaded_files)
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)
        embeddings = load_embeddings(text_chunks)

        # Create vector store
        vector_store = FAISS.from_documents(embeddings, embedding=None)

        # Create the chain object
        chain = create_conversational_chain(vector_store)
        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
