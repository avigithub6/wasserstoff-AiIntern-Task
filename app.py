import streamlit as st
import os
from pathlib import Path
from backend.document_processor.processor import DocumentProcessor
from utils.helpers import display_results, process_file_upload
from dotenv import load_dotenv

# Set page configuration first
st.set_page_config(
    page_title="Document Research & Theme Identification",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Debug: Print API key (we'll remove this later)
st.write("API Key loaded:", bool(os.getenv("OPENAI_API_KEY")))

# Initialize document processor
@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

def main():
    # Initialize processor
    processor = get_document_processor()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    process_file_upload(uploaded_file, processor)
                st.success("Documents processed successfully!")
    
    # Main content area
    st.title("Document Research & Theme Identification Chatbot")
    
    # Query input
    user_query = st.text_input("Ask a question about your documents:")
    
    if user_query:
        with st.spinner("Analyzing documents..."):
            # Get individual document answers
            doc_answers = processor.get_document_answers(user_query)
            
            # Get theme-based synthesis
            themes = processor.identify_themes(doc_answers)
            
            # Display results
            display_results(doc_answers, themes)
    
    # Display document statistics
    with st.sidebar:
        st.subheader("Document Statistics")
        stats = processor.get_stats()
        st.write(f"Total Documents: {stats['total_docs']}")
        st.write(f"Processed Pages: {stats['total_pages']}")

if __name__ == "__main__":
    main() 