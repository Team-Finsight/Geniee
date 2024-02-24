
from flask import Flask, request, jsonify
from PIL import Image
# LangChain Imports
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain.chat_models import ChatAnyscale
from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# Flask Extensions
from flask_cors import CORS
import fitz
import logging
from pythonjsonlogger import jsonlogger

api_key = 'esecret_sn18dj7defst1n3aacssvnc94m'
api_base = "https://api.endpoints.anyscale.com/v1"

def chat(query):
    if not query:
        return jsonify({'error': 'No query provided'})
    try:
        from langchain import PromptTemplate,  LLMChain
        template = "Chat about {text}"
        prompt = PromptTemplate(template=template, input_variables=["text"])
        llm = ChatAnyscale(anyscale_api_key=api_key, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1")
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run(text=query)
        return response
    except Exception as e:
        return print(e)

