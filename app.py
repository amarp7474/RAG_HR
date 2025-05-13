import os
import numpy as np
import streamlit as st
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import retrieval_qa
##os.chdir("C:\\Users\\ADMIN\\OneDrive\\Desktop\\RAG_kirsh")

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embedings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    data_load=PyPDFLoader("HR_Policy_Manual_2023.pdf")
    load=data_load.load()
    print(load)

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    docs=text_splitter.split_documents(load)
    return docs

def vector_store(docs, path="faiss_index"):
    vectorstore = FAISS.from_documents(docs, bedrock_embedings)
    vectorstore.save_local("C:\\Users\\ADMIN\\OneDrive\\Desktop\\RAG_kirsh")

def get_claude():
    llm=BedrockLLM(model_id="anthropic.claude-v2:1",   client= bedrock, model_kwargs=({'maxTokens:512'}))
    return llm

def get_llama2():
    llm=BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0",   client= bedrock, model_kwargs=({'max_gen_len : 512'}))
    return llm


prompt_template="""
    Human: following piuece of context to provide a concise answer to the question at the end but usse atleast
    summarize with 250 words with detailed explanations. if dont know the answer just say dont know. dont try 
    to make up answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vector_store, query):
    qa= retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True, 
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer=qa({"query":query})
    return(answer['result'])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF to Get Answers")

    user_question = st.text_input("Ask a question")

    with st.sidebar:
        st.title("Menu")

        if st.button("Vectors update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                vector_store(docs, path="faiss_index")
                st.success("Vector store created!")

    if st.button("Claude output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embedings)
            llm = get_claude()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

    if st.button("Llama2 output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embedings)
            llm = get_llama2()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")

if __name__=="__main__":
    main()