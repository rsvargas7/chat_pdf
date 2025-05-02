import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# App title and presentation
st.title('Generación asistida por recuperación de información')
st.write("Edición de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('B.jpg')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"La imagen no se ha podido mostrar: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("Este asistente te apoyará en el análisis del PDF que subiste")

# Get API key from user
ke = st.text_input('Introduce tu API key de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Introduce tu clave API de OpenAI para seguir adelante, por favor.")

# PDF uploader
pdf = st.file_uploader("Sube el archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto recuperado: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento segmentado en {len(chunks)} fragmentos")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("Indica qué información te gustaría obtener del documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
        
        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use a current model instead of deprecated text-davinci-003
            # Options: "gpt-3.5-turbo-instruct" or "gpt-4-turbo-preview" depending on your API access
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Hubo un problema al procesar el PDF: {str(e)}")
        # Add detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Introduce tu clave API de OpenAI para continuar, por favor.")
else:
    st.info("Sube un archivo PDF para empezar.")
