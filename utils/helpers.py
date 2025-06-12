import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import time

def process_file_upload(uploaded_file, processor):
    """Process an uploaded file and add it to the document processor."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        # Write uploaded file content to temp file
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Process the document
        processor.process_document(tmp_path, uploaded_file.name)
    finally:
        # Clean up with retry mechanism
        max_retries = 3
        for i in range(max_retries):
            try:
                Path(tmp_path).unlink()
                break
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(0.5)  # Wait before retrying
                continue

def display_results(doc_answers, themes):
    """Display document answers and identified themes."""
    # Display individual document answers
    st.subheader("Document Answers")
    
    if doc_answers:
        # Create a DataFrame for document answers
        df = pd.DataFrame(doc_answers)
        df.columns = ["Document ID", "Extracted Answer", "Citation"]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No relevant answers found in the documents.")
    
    # Display themes
    st.subheader("Identified Themes")
    
    if themes:
        for i, theme in enumerate(themes, 1):
            with st.expander(f"Theme {i}: {theme['name']}", expanded=True):
                st.write("**Summary:**")
                st.write(theme['summary'])
                st.write("**Supporting Documents:**")
                for doc in theme['documents']:
                    st.write(f"- {doc['id']}: {doc['citation']}")
    else:
        st.info("No themes identified from the current answers.") 