import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import os

# Configuración de la página
st.set_page_config(page_title="Consulta RTE", layout="wide")
st.title("Consulta Reglamentos Técnicos Ecuatorianos")

# Inicializar variables de sesión
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = False

def process_files():
    text = ""
    # Lee los archivos TXT de la carpeta RTE_Procesados
    for file in os.listdir("RTE_Procesados"):
        if file.endswith(".txt"):
            with open(os.path.join("RTE_Procesados", file), 'r', encoding='utf-8') as f:
                text += f.read() + "\n\n"
    
    # Divide el texto en fragmentos
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Crear embeddings utilizando un modelo más robusto
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Crear base de datos vectorial
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

# Interfaz principal
if not st.session_state.processed_files:
    with st.spinner("Procesando archivos RTE..."):
        try:
            knowledge_base = process_files()
            st.session_state.knowledge_base = knowledge_base
            st.session_state.processed_files = True
            st.success("¡Archivos procesados correctamente!")
        except Exception as e:
            st.error(f"Error al procesar archivos: {str(e)}")

# Área de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Haga su consulta sobre los RTE"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Buscar más documentos relevantes para mejorar las respuestas
            docs = st.session_state.knowledge_base.similarity_search(prompt, k=5)
            
            # Crear la cadena de QA
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
            )
            chain = load_qa_chain(llm, chain_type="stuff")

            # Aumentar la longitud máxima de respuesta
            response = chain.run(input_documents=docs, question=prompt, max_length=500, temperature=0.7)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")

