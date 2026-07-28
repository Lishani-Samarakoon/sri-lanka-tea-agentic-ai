import os
import streamlit as st

from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_documents():
    documents = []
    folder_path = "annual_reports"

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)
    return docs


@st.cache_resource
def build_vectorstore():
    docs = load_documents()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


def get_retriever():
    vectorstore = build_vectorstore()
    return vectorstore.as_retriever()