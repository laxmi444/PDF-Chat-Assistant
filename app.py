# RAG Q&A chatbot with pdf including chat history
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# setting up streamlit app
# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant", 
    page_icon="📄", 
    #layout="wide"
)
st.title("🤖 Conversational PDF Assistant")
st.write("Upload PDFs and Chat Intelligently!")

# input groq api key
api_key=st.text_input("Enter your Groq API Key:",type="password")

# checking if api is provided
if api_key:
    llm=ChatGroq(model_name="gemma2-9b-it",groq_api_key=api_key)
    
    # chat interface
    session_id=st.text_input("Session ID", value="default_session")

    # statefully manage chat history
    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose a PDF file to upload", type="pdf", accept_multiple_files=True)

    # process uploaded pdfs
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f"temp_{uploaded_file.name}"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            os.unlink(temppdf)  # Clean up temporary file

    # splitting
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        import chromadb
        from chromadb.config import Settings

        # Add this before vectorstore creation
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db", 
            settings=Settings(allow_reset=True)
        )

        # Modify Chroma initialization
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            client=chroma_client,  # Add this line
            collection_name="pdf_documents"  # Optional: specify collection name
        )
        retriever=vectorstore.as_retriever()

# prompt
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question,"
            "just reformulate it if needed and otherwise return it as is."

        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        # answer question
        system_prompt=(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. "
            "\n\n"
            "{context}"  # Remove the space after {context
)


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_message_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer" 
        )
        user_input=st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                    #constructs a key "abcd123 in "store"
                },
            )
            st.write(st.session_state.store)
            st.success("Assistant: " + response["answer"])
            st.write("Chat history:", session_history.messages)
else:
    st.warning("Please enter Groq API key.")