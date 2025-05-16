import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
from langchain.schema import Document # Schema Created in backend
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

  #API Configuration
  
genai.configure(api_key=os.getenv("GOOGLE_API_1"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

## cashe the HF embeddings to avoid slow reload of the embeddings
@st.cache_resource(show_spinner="Loading Embedding Model...")

def embeddings():
    return(HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2"))

embedding_model = embeddings()

## user interface


st.header("Rag Assitant:HF Embeddings + Gemini LLM")
st.subheader("Your AI Doc Assistant")

uploaded_file = st.file_uploader(label = "Upload the PDF Doc",type = "pdf")

if uploaded_file:
    raw_text =""
    pdf = PdfReader(uploaded_file)
    for i , page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text+=text
            
    if raw_text.strip():
        document = Document(page_content=raw_text)
        ## using chartextsplitter we will create chunks and pass it into the model
        
        splitter=CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunks = splitter.split_documents([document])
        
        ## store the chunks into the FAISS Vectordb
        chunk_pieces = [chunk.page_content for chunk in chunks]
        vectordb = FAISS.from_texts(chunk_pieces, embedding_model)
        retriever =  vectordb.as_retriever() # Retrieve the vectors..
        
        st.success("Embeddings are Generated. Ask your question!")
        
        user_input = st.text_input(label = "Enter your question")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.spinner("Analyzing the Document"):
                relevant_docs = retriever.get_relevant_documents(user_input)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                prompt = f''' you are an expert assistant. use the context below to answer the query.If unsure or
                information not avaiable in the doc, pass the message - "Information is not there. Look into
                other sources." 
                context : {context},
                query : {user_input}, 
                Answer: '''
                
                response = gemini_model.generate_content(prompt)
                st.markdown("Answer: ")
                st.write(response.text)
    else:
        st.warning("No Text could be Extracted from PDF. Please upload a readable PDF")
        
            
        
        
        