import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Page Config
st.set_page_config(page_title="BharatInvest-Agent", layout="wide")

st.title("🚀 BharatInvest-Agent")
st.markdown("*Dismantling high-friction bottlenecks in Indian Retail Investing.*")

# 1. Technical Specificity: API Setup
# Use st.secrets for secure deployment
try:
    api_key = st.secrets["gemini_api_key"]
except KeyError:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 2. RAG Architecture: Grounding in Verified Data
    uploaded_file = st.file_uploader("Upload a SEBI DRHP or Annual Report (PDF)", type="pdf")
    
    if uploaded_file:
        # Use temporary file for Streamlit Cloud compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_pdf_path = temp_file.name
        
        # Load and split document
        loader = PyPDFLoader(temp_pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        # Vector Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # 3. Agentic Workflow
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )
        
        query = st.text_input("Ask about Risk Factors, Revenue Trends, or P/E Ratios:")
        
        if query:
            with st.spinner("Analyzing high-friction bottlenecks..."):
                response = qa_chain.invoke(query)
                st.subheader("Analysis Results")
                st.write(response["result"])
                
                # Ethical Consideration & Transparency
                st.caption("Note: This analysis is based strictly on the uploaded document to prevent hallucinations.")
        
        # Clean up temp file after processing
        os.unlink(temp_pdf_path)
else:
    st.warning("Please enter your Gemini API Key in the sidebar to begin.")