import os
import numpy as np
import streamlit as st
import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import  PromptTemplate
from langchain.chains import retrieval_qa
from langchain_aws import BedrockLLM


bedrock=boto3.client(service_name='bedrock-runtime')
Bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


def data_ingestion():
    loader=PyPDFDirectoryLoader("APP_RAG")
    documents=loader.load()
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs=text_splitter.split_text(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs, Bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")


def get_claude_llm():
    llm=BedrockLLM(model_id="anthropic.claude-v2:1", client=bedrock, model_kwargs=({"max_tokens":512}))
    return llm

def get_llama2_llm():
    llm=BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs= ({"max_gen_len":512}))
    return llm

prompt_template="""
    Human: following piuece of context to provide a concise answer to the question at 
    the end but usse atleast summarize with 250 words with detailed explanations. if dont 
    know the answer just say dont know. dont try to make up answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant: """

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):
    qa= retrieval_qa.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k":3}
        ),
        return_source_documents=True, 
        chain_type_kwargs={"prompt":PROMPT}
    )
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    st.header("chat with SOP")

    with st.sidebar:
        st.title("Menu")
        if st.button("update vector"):
            with st.spinner("processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("done")
    user_quesyion=st.text_input("ask a question:")
    if st.button("claude_output"):
        with st.spinner("processing"):
            faiss_index=FAISS.load_local("faiss_index", Bedrock_embeddings)
            llm=get_claude_llm()

            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("done")

if __name__=="__main__":
    main()




