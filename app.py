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
    documents = []
    # Lee los archivos TXT de la carpeta RTE_Procesados
    for file in os.listdir("RTE_Procesados"):
        if file.endswith(".txt"):
            with open(os.path.join("RTE_Procesados", file), 'r', encoding='utf-8') as f:
                content = f.read()
                # Extraer el nombre del RTE del archivo
                rte_name = file.split('.')[0]  # Asume que el archivo se llama "RTE_030.txt"
                # Agregar metadata al texto
                documents.append({"content": content, "source": rte_name})
    
    # Divide el texto usando un splitter más específico
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,  # Reducimos el tamaño para mantener contexto más específico
        chunk_overlap=100,
        length_function=len
    )

    chunks = []
    for doc in documents:
        texts = text_splitter.split_text(doc["content"])
        # Agregar metadata a cada chunk
        for text in texts:
            # Asegurarse de que los títulos importantes se mantengan
            if "1. OBJETO" in text or "2. CAMPO DE APLICACION" in text:
                chunk_size = 800  # Chunks más grandes para secciones importantes
            chunks.append({
                "content": text,
                "source": doc["source"],
                "is_title": "1. OBJETO" in text or "2. CAMPO DE APLICACION" in text
            })
    
    # Crear embeddings con modelo multilingüe optimizado
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # Modelo más potente
        model_kwargs={'device': 'cpu'}
    )
    
    # Crear base de datos vectorial con metadatos
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "is_title": chunk["is_title"]} for chunk in chunks]
    
    knowledge_base = FAISS.from_texts(
        texts, 
        embeddings, 
        metadatas=metadatas
    )
    
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
            # Buscar documentos relevantes
            docs = st.session_state.knowledge_base.similarity_search(prompt)
            
            # Crear la cadena de QA
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Generar respuesta
            response = chain.run(input_documents=docs, question=prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")
