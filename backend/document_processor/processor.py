import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pytesseract
from pypdf import PdfReader
import magic
from PIL import Image
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import json

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with necessary components."""
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(1536)  # OpenAI embeddings are 1536 dimensions
        self.documents = []
        self.doc_stats = {"total_docs": 0, "total_pages": 0}
    
    def process_document(self, file_path: str, original_filename: str) -> None:
        """Process a document and add it to the vector store."""
        # Detect file type
        file_type = magic.from_file(file_path, mime=True)
        
        # Extract text based on file type
        if file_type == "application/pdf":
            text = self._process_pdf(file_path)
        elif file_type.startswith("image/"):
            text = self._process_image(file_path)
        elif file_type.startswith("text/"):
            text = Path(file_path).read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Split text into chunks
        docs = self.text_splitter.create_documents([text])
        
        # Update document statistics
        self.doc_stats["total_docs"] += 1
        self.doc_stats["total_pages"] += len(docs)
        
        # Process each chunk
        for i, doc in enumerate(docs):
            # Create document metadata
            metadata = {
                "doc_id": f"DOC{len(self.documents):03d}",
                "original_filename": original_filename,
                "page_number": i + 1,
                "citation": f"Page {i + 1}"
            }
            
            # Get embeddings
            embedding = self.embeddings.embed_query(doc.page_content)
            
            # Add to FAISS index
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Store document
            self.documents.append({
                "content": doc.page_content,
                "metadata": metadata
            })
    
    def _process_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _process_image(self, file_path: str) -> str:
        """Extract text from an image using OCR."""
        try:
            # Open and verify the image using PIL
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Perform OCR
                text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")
    
    def get_document_answers(self, query: str) -> List[Dict[str, str]]:
        """Get relevant answers from documents for a given query."""
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in FAISS
        k = min(5, len(self.documents))  # Get top 5 or all if less than 5
        D, I = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        
        # Prepare context from relevant documents
        context = ""
        relevant_docs = []
        for idx in I[0]:
            doc = self.documents[idx]
            context += f"\nDocument {doc['metadata']['doc_id']}:\n{doc['content']}\n"
            relevant_docs.append(doc)
        
        # Generate answers using LLM
        prompt = f"""Based on the following documents, answer the question: {query}

Documents:
{context}

For each relevant document, provide:
1. The specific answer found in that document
2. The exact citation (document ID and page/paragraph)

Format the answer as a list of JSON objects with fields:
- doc_id: Document ID
- answer: Extracted answer
- citation: Page/paragraph citation"""
        
        response = self.llm.invoke(prompt)
        
        # Parse and format answers
        try:
            answers = json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
            answers = [
                {
                    "doc_id": doc["metadata"]["doc_id"],
                    "answer": "Could not extract structured answer",
                    "citation": doc["metadata"]["citation"]
                }
                for doc in relevant_docs
            ]
        
        return answers
    
    def identify_themes(self, doc_answers: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Identify common themes across document answers."""
        if not doc_answers:
            return []
        
        # Prepare context from answers
        context = "\n".join([
            f"Document {ans['doc_id']}: {ans['answer']}"
            for ans in doc_answers
        ])
        
        prompt = f"""Analyze the following document answers and identify common themes:

{context}

Identify 2-3 main themes and provide:
1. Theme name
2. Theme summary
3. Supporting documents with citations

Format the response as a list of JSON objects with fields:
- name: Theme name
- summary: Theme summary
- documents: List of supporting document objects with id and citation"""
        
        response = self.llm.invoke(prompt)
        
        try:
            themes = json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
            themes = [{
                "name": "Theme Analysis Failed",
                "summary": "Could not extract themes from the answers",
                "documents": [
                    {
                        "id": ans["doc_id"],
                        "citation": ans["citation"]
                    }
                    for ans in doc_answers
                ]
            }]
        
        return themes
    
    def get_stats(self) -> Dict[str, int]:
        """Get document processing statistics."""
        return self.doc_stats 