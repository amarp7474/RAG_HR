
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM
import boto3
ession = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_DEFAULT_REGION"]
)

client = session.client(service_name='bedrock-runtime')

def hr_index():
    
    data_load=PyPDFLoader('LifeInsuranceAndAnnuities-2.pdf')  
    
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100,chunk_overlap=10)
    
    data_embeddings=BedrockEmbeddings(
    credentials_profile_name= 'default',
    region_name='us-east-1',
    model_id='amazon.titan-embed-text-v1')
    
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    
    db_index=data_index.from_loaders([data_load])
    return db_index

def hr_llm():
    llm=BedrockLLM(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
        "max_tokens_to_sample":3000,
        "temperature": 0.1,
        "top_p": 0.9})
    return llm

def hr_rag_response(index,question):
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query
#  https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html


import streamlit as st
##import rag_backend as demo

st.set_page_config(page_title="HR QA and answers using RAG")
st.markdown("<h1 style='text-align: center;'>HR Q&A with RAG</h1>", unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner('Loading documents and building index...'):
        st.session_state.vector_index = hr_index()

input_text = st.text_area('Ask your question here:', label_visibility='collapsed')
go_button = st.button("Submit")

if go_button:
    if not input_text.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            response = hr_rag_response(index=st.session_state.vector_index, question=input_text)
            st.write(response)
