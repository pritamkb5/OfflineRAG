import os
import uuid
from io import BytesIO
import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st

# --- Page Configuration (MUST BE FIRST STREAMLIT CALL) ---
st.set_page_config(page_title="Local PDF RAG with Ollama & Qdrant", layout="wide")

# --- Configuration ---
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# --- Initialization ---
@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(path=QDRANT_PATH)
    return client

client = get_qdrant_client()

# Ensure collection exists
try:
    client.get_collection(collection_name=COLLECTION_NAME)
except Exception:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text: str):
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def process_pdf(file):
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(file)
    
    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)
    
    with st.spinner("Generating embeddings and storing in Qdrant (this may take a while)..."):
        points = []
        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "source": file.name}
            )
            points.append(point)
            
            # Batch upload to avoid memory issues on large docs
            if len(points) >= 50:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []
                
        # Upload remaining
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            
    st.success(f"Successfully processed and stored {file.name}")
    return text

def retrieve_context(query: str, top_k: int = 3):
    query_vector = get_embedding(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    context = "\n\n".join([hit.payload["text"] for hit in results])
    return context

def ask_ollama(query: str, context: str):
    prompt = f"""Use the following context to answer the question. If you don't know the answer, just say that you don't know.

Context:
{context}

Question:
{query}

Answer:"""
    
    response = ollama.chat(model=LLM_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def generate_summary_pdf(summary_text: str):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []
    
    title = Paragraph("Document Summary Report", styles['Title'])
    flowables.append(title)
    flowables.append(Spacer(1, 12))
    
    for para in summary_text.split('\n'):
        if para.strip():
            p = Paragraph(para.strip(), styles['BodyText'])
            flowables.append(p)
            flowables.append(Spacer(1, 6))
            
    doc.build(flowables)
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.title("📄 Local PDF RAG App")
st.markdown("100% Offline-local document Q&A using Streamlit, Qdrant, Ollama (Llama 3), and ReportLab.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process Document"):
            full_text = process_pdf(uploaded_file)
            st.session_state.full_text = full_text
            st.session_state.messages = []  # Clear history on new doc
            
    st.header("Generate Report")
    if st.session_state.full_text:
        if st.button("Generate Summary Report"):
            with st.spinner("Generating summary with Llama 3..."):
                context = retrieve_context("What is this document about? Summarize the key points.", top_k=5)
                summary = ask_ollama("Please write a comprehensive summary based on the provided context.", context)
                
                pdf_buffer = generate_summary_pdf(summary)
                
                st.download_button(
                    label="Download Summary PDF",
                    data=pdf_buffer,
                    file_name="summary_report.pdf",
                    mime="application/pdf"
                )

# Chat Interface
st.header("Semantic Search & QA")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if query := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = retrieve_context(query)
            response = ask_ollama(query, context)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})