import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os
import langchain
from time import sleep
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

st.title("News Research Tool")

st.sidebar.title("News Articles URLs")
main_placeholder = st.empty()
file_path = "vector_index.pkl"
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    main_placeholder.text("URL Loading...")
    if url.strip():
        urls.append(url)

process_button = st.sidebar.button("Process URL")

if process_button:
    loader = UnstructuredURLLoader(urls)
    data = loader.load()

    if not data:
        st.error("Failed to load content from the provided URLs.")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=500
        )

        main_placeholder.text("Text Splitter Starting...")

        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

        vectorindex_hf = FAISS.from_documents(docs, embeddings)

        main_placeholder.text("Vector Building Started...")
        sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_hf, f)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain.invoke({"question": query})

            st.header("Result")
            st.write(result["answer"])

            sources = result.get("sources", " ")
            if sources:
                st.subheader("Sources: ")
                for source in sources.split("\n"):
                    st.write(source)





