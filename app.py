import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDF", page_icon=" :books:")

    st.header("Chat with multiple PDF")
    st.text_input("Ask question about your PDF")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDFs"):
                #* get pdf texts
                raw_text = get_pdf_text(pdf_docs)

                #* get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #* create vector store
                vectorsore = get_vectorstore(text_chunks)

if __name__ == '__main__':
    main()