# imports
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

if 'upload_page' not in st.session_state:
    st.session_state['upload_page'] = True
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = None
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = None
if 'llm' not in st.session_state:
    st.session_state['llm'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your PDF"):
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role":"user", "content":prompt})
        pdf_context = ""
        messages_context = ""
        retriever = st.session_state['vector_db'].as_retriever(search_type="mmr", search_kwargs={"k": 3})
        relevant_pdfs = retriever.get_relevant_documents(prompt)
        for pdf in relevant_pdfs:
            pdf_context += pdf.page_content + "\n"
        # Only use recent user messages, not assistant. so that bot cant use previous answers as context, it causes misinformation
        recent_messages = st.session_state['messages'][::-2][:4]
        for entry in recent_messages:
            messages_context += entry["role"] + ": "
            messages_context += entry["content"] + "\n"
        chat_prompt = f"""
        As an AI assistant, your task is to provide accurate and relevant responses to my queries based on the content of the uploaded documents. While the chat history provides context for our conversation, it should not override the factual information from the documents. For each query, carefully consider the document context as the primary source of information.

        Previous Conversation:
        {messages_context}

        Document Context:
        {pdf_context}

        Now, I am asking you this: {prompt}.

        In your response, prioritize the information from the documents. If the chat history helps in understanding the context or nuances of the query, use it to enhance the response, but do not let it contradict or replace the facts from the documents. Provide a clear and accurate answer based on this guidance.
        """
        response = st.session_state['llm'](chat_prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state['messages'].append({"role":"assistant", "content":response})

def vectorize():
    # Turning pdf files into PdfReader objects
    pdf_readers = []
    for uploaded_file in st.session_state['uploaded_files']:
        pdf_readers.append(PdfReader(uploaded_file))

    # Extracting text from pdfs and concatenating
    text = ""
    for pdf in pdf_readers:
        for page in pdf.pages:
            text += page.extract_text()

    # Larger Chunks -> Faster Splitting -> Less Embedding Iterations -> Possibly Worse Performance
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=32)
    chunks = text_splitter.split_text(text)

    # Embedding each chunk and storing inside vector database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state['vector_db'] = Chroma.from_texts(chunks, embeddings)

st.sidebar.title("About")
st.sidebar.write("""This application uses the Llama2-Chat Large Language Model
                 in conjunction with Chroma Vector Database to provide
                 Retrieval-Augmented Generation. This allows the LLM to get context
                 from your PDF documents in order to answer your queries, instead of having
                 to be directly fine tuned (which is extremely compute expensive).""")
st.sidebar.caption(":blue[_Made by Jeffrey Gordon_]")
libraries = [
    "Streamlit",
    "Langchain",
    "Replicate",
    "ChromaDB",
    "Llama2-Chat",
    "HuggingFaceEmbeddings"
]

topics = [
    "Word Embedding",
    "Vector Databases",
    "Document Retrieval",
    "LLM Inference",
    "Prompt Engineering",
    "Retrieval-Augmented Generation",
    "LLM RAG - Speed & Performance Optimization"
]

list_items = "\n".join([f"- {library}" for library in libraries])
st.sidebar.subheader("Libraries Used:")
st.sidebar.markdown(list_items)
topic_items = "\n".join([f"- {topic}" for topic in topics])
st.sidebar.subheader("Topics Covered:")
st.sidebar.markdown(topic_items)

st.title("PDF AI Chat")
placeholder = st.empty()
if st.session_state['upload_page']:
    with placeholder.form("my_form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload your PDFs:", type='pdf', accept_multiple_files=True)
        submitted = st.form_submit_button("Submit")
        if submitted and not uploaded_files:
            st.write("Must Upload a PDF! Try Again!")
        elif submitted:
            st.session_state['uploaded_files'] = uploaded_files
            st.session_state['upload_page'] = False

if not st.session_state['upload_page']:
    placeholder.empty()
    if not st.session_state['vector_db']:
        with st.spinner("Putting your documents in a Vector Database..."):
            vectorize()
            st.session_state['llm'] = Replicate(model="meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",
                            model_kwargs={"temperature": 0.75, "max_new_tokens":1024, "top_p": 1,},)
    if st.sidebar.button("Start Over", type="primary"):
        st.session_state['upload_page'] = True
        st.session_state['uploaded_files'] = None
        st.session_state['vector_db'] = None
        st.session_state['messages'] = []
        st.rerun()
    chat()