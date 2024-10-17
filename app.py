import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os

# Configuración de la página
st.set_page_config(page_title="Consulta RTE", layout="wide")
st.title("Consulta Reglamentos Técnicos Ecuatorianos")

# Inicializar variables de sesión
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = False

def process_files():
    files = []
    # Lee los archivos TXT de la carpeta RTE_Procesados
    for file in os.listdir("RTE_Procesados"):
        if file.endswith(".txt"):
            with open(os.path.join("RTE_Procesados", file), 'r', encoding='utf-8') as f:
                files.append(f.read())
    
    # Divide el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.create_documents(files)
    
    # Crear embeddings
    embeddings = HuggingFaceEmbeddings()
    
    # Crear base de datos vectorial
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectordb"
    )
    
    return vectordb

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain

# Interfaz principal
if not st.session_state.processed_files:
    st.info("Procesando archivos RTE...")
    vectorstore = process_files()
    st.session_state.conversation = get_conversation_chain(vectorstore)
    st.session_state.processed_files = True
    st.success("¡Archivos procesados correctamente!")

# Área de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Haga su consulta sobre los RTE"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.session_state.conversation({'question': prompt})
        st.markdown(response['answer'])
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
