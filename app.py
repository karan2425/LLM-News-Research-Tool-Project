import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.qa_with_sources.retrieval import (
    RetrievalQAWithSourcesChain
)

load_dotenv()

st.set_page_config(
    page_title="Stock Intelligence Q&A",
    layout="wide"
)

st.title("📊 Stock Intelligence Q&A")
st.write("Ask questions about Nvidia & HDFC Bank using live financial data")

with st.sidebar:
    st.header("🔄 Data Ingestion")
    ingest_btn = st.button("Ingest Data")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8,
)

@st.cache_resource(show_spinner=True)
def ingest_data():
    urls = [
        "https://www.moneycontrol.com/us-markets/stockpricequote/nvidia/NVDA",
        "https://www.moneycontrol.com/financials/hdfcbank/balance-sheetVI/HDF01"
    ]

    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000,
        chunk_overlap=0
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=os.environ["INDEX_NAME"]
    )

    return vectorstore

if ingest_btn:
    with st.spinner("Ingesting data..."):
        vectorstore = ingest_data()
    st.sidebar.success("✅ Data ingestion complete!")

st.header("❓ Ask a Question")

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What is Nvidia?"
)

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            vectorstore = ingest_data()

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=retriever
            )

            result = chain.invoke({"question": query})

        st.subheader("✅ Answer")
        st.write(result["answer"])

        st.subheader("🔗 Sources")
        st.write(result["sources"])
