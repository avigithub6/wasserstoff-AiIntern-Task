# Document Research & Theme Identification Chatbot

A powerful document analysis system that uses AI to process documents, answer questions, and identify themes across multiple documents.

## Features

- Document Upload & OCR Processing
  - Support for PDF, Images, and Text files
  - OCR using PaddleOCR for scanned documents
  - Vector storage using FAISS

- Natural Language Querying
  - Ask questions in plain English
  - Get precise answers with citations
  - Document-level and sentence-level granularity

- Theme Identification
  - Automatic theme detection across documents
  - Synthesized answers with multi-document citations
  - Theme-based document clustering

## Tech Stack

- Frontend: Streamlit
- Backend: FastAPI
- OCR: PaddleOCR
- Vector Store: FAISS
- LLM: OpenAI
- Document Processing: langchain

## Setup Instructions

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
chatbot_theme_identifier/
├── app.py                 # Main Streamlit application
├── backend/
│   ├── document_processor.py
│   ├── ocr_handler.py
│   ├── vector_store.py
│   └── theme_analyzer.py
├── utils/
│   ├── text_extraction.py
│   └── helpers.py
├── data/                  # Document storage
├── requirements.txt
└── README.md
```

## Usage

1. Upload Documents:
   - Use the upload interface to add documents
   - Supported formats: PDF, PNG, JPG, TXT

2. Ask Questions:
   - Type natural language questions
   - View answers with source citations
   - Explore identified themes

3. Theme Analysis:
   - View automatically identified themes
   - See theme-based document groupings
   - Access synthesized answers

## License

MIT License 