import streamlit as st
from PIL import Image
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
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

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! This is a Contracts Evaluation Tool"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Conversation function
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Display chat history
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Looking into the files...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Create conversational chain
def create_conversational_chain(vector_store):
    # Replace with your Replicate or other language model initialization code
    llm = Replicate(
        streaming=True,
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

# Specific prompts after file upload
def perform_specific_prompts(chain):
    specific_questions = [
        "Does the uploaded document considered sustainable contract?",
        "How to make it a sustainable contract?"
    ]

    for question in specific_questions:
        st.write(f"Question: {question}")
        answer = conversation_chat(question, chain, st.session_state['history'])
        message(question, is_user=True, key=question + '_question', avatar_style="thumbs")
        message(answer, key=question, avatar_style="fun-emoji")
        st.session_state['past'].append(question)
        st.session_state['generated'].append(answer)

# Main function
def main():
    initialize_session_state()
    st.title("Evaluation tool for contracts sustainability")
    st.write('This project was Developed by group 1 as showcase for 602 - Project procurement course')
    
    # Sidebar for file upload
    st.sidebar.title("Please drop your contract documents here")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    # Display images in sidebar
    image = Image.open('./logo/alberta-logo-university-com.png')
    st.sidebar.image(image, use_column_width=True)
    image = Image.open('./logo/Faculty_Wordmark_Standard.png')
    st.sidebar.image(image, use_column_width=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            else:
                continue

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        chain = create_conversational_chain(vector_store)

        # Perform specific prompts
        perform_specific_prompts(chain)

        # Display chat history
        display_chat_history(chain)

if __name__ == "__main__":
    main()
