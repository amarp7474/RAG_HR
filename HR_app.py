import os
import numpy as np
import streamlit as st
import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
from langchain_aws import BedrockLLM


# Initialize the AWS Bedrock client
bedrock = boto3.client(service_name='bedrock-runtime')

# Set up embeddings for the text using Bedrock
Bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Load PDF files from the specified directory
def data_ingestion():
    loader = PyPDFDirectoryLoader("C:/Users/ADMIN/OneDrive/Desktop/RAG_kirsh/App_RAG")
    documents = loader.load()
    
    # Combine text from all documents into a single string
    full_text = ""
    for doc in documents:
        full_text += doc.page_content  # assuming each document has a `page_content` attribute
    
    # Split the full text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_text(full_text)  # Now we pass a single string to split_text
    
    return docs


# Set up FAISS vector store from the documents
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, Bedrock_embeddings)
    return vectorstore_faiss

# Get Claude LLM (language model from Bedrock)
def get_claude_llm():
    llm = BedrockLLM(model_id="anthropic.claude-v2:1", client=bedrock, model_kwargs={"max_tokens": 512})
    return llm

# Set up prompt template
prompt_template = """
    Human: following piece of context to provide a concise answer to the question at 
    the end but use at least summarize with 250 words with detailed explanations. If you don't 
    know the answer, just say "don't know". Don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to get the response from the model
def get_response_llm(llm, vectorstore_faiss, query):
    qa = retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.header("Chat with HR Policy and Annuity PDF Documents")

    with st.sidebar:
        st.title("Menu")
        if st.button("Update Vector Store"):
            with st.spinner("Processing..."):
                docs = data_ingestion()  # Ingest the PDFs and prepare them
                vector_store = get_vector_store(docs)  # Get the vector store
                st.success("Vector store updated successfully!")

    user_question = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            # Load the vector store and LLM
            vectorstore_faiss = get_vector_store(data_ingestion())
            llm = get_claude_llm()

            # Get the response for the user's question
            response = get_response_llm(llm, vectorstore_faiss, user_question)
            st.write(response)
            st.success("Done!")

if __name__ == "__main__":
    main()
