from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st


# Function for API configuration at sidebar
def sidebar_api_key_configuration():
    st.sidebar.subheader("API Keys")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key 🗝️", type="password",
                                           help='Get OpenAI API Key from: https://platform.openai.com/api-keys')
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key 🗝️", type="password",
                                         help='Get Groq API Key from: https://console.groq.com/keys')
    if openai_api_key == '' and groq_api_key == '':
        st.sidebar.warning('Enter the API Key(s) 🗝️')
        st.session_state.prompt_activation = False
    elif (openai_api_key.startswith('sk-') and (len(openai_api_key) == 56)) and (groq_api_key.startswith('gsk_') and
                                                                                 (len(groq_api_key) == 56)):
        st.sidebar.success('Lets Proceed!', icon='️👉')
        st.session_state.prompt_activation = True
    else:
        st.sidebar.warning('Please enter the correct API Key 🗝️!', icon='⚠️')
        st.session_state.prompt_activation = False
    return openai_api_key, groq_api_key


def sidebar_groq_model_selection():
    st.sidebar.subheader("Model Selection")
    model = st.sidebar.selectbox('Select the Model', ('Llama3-8b-8192', 'Llama3-70b-8192', 'Mixtral-8x7b-32768',
                                                      'Gemma-7b-it'), label_visibility="collapsed")
    return model


# Read PDF data
def read_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split data into chunks
def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(text)
    return text_chunks


# Create vectorstore
def create_vectorstore(openai_api_key, pdf_docs):
    raw_text = read_pdf_data(pdf_docs)  # Get PDF text
    text_chunks = split_data(raw_text)  # Get the text chunks
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Get response from llm of user asked question
def get_llm_response(llm, prompt, question):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.vector_store.as_retriever(), document_chain)
    response = retrieval_chain.invoke({'input': question})
    return response
