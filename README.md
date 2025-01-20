# KnowledgeBasePDFAssistant

The provided code is a **FastAPI** application that serves as a knowledge base using PDF documents.

## Dependencies

The project relies on the following libraries and tools:

- **chromadb**: A database client for storing and querying vector embeddings.
- **fitz (PyMuPDF)**: To extract text from PDF files.
- **uvicorn**: An ASGI server to run the FastAPI application.
- **fastapi**: A modern web framework for building APIs.
- **langchain.text_splitter**: To split large texts into smaller chunks.
- **openai**: To interact with OpenAI's API, specifically Azure's version.
- **pydantic**: For data validation and settings management through Python dataclasses.
- **sentence_transformers**: To generate embeddings for text chunks.

### Running the Application

Use the following command to run the FastAPI application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Flow Diagram

```python
+-------------------+
| Start             |
+-------------------+
         |
         v
+-------------------+
| Import Libraries  |
+-------------------+
         |
         v
+-------------------+
| Initialize App    |
| & Global Variables|
+-------------------+
         |
         v
+-------------------+
| Define Functions: |
| - extract_text    |
| - preprocess_text |
| - index_pdf_text  |
| - query_knowledge |
+-------------------+
         |
         v
+-------------------+
| Define Pydantic   |
| Model: QueryRequest|
+-------------------+
         |
         v
+-------------------+
| Define API        |
| Endpoint: /query  |
+-------------------+
         |
         v
+-------------------+
| Main Block:       |
| - Index PDFs      |
| - Run Server      |
+-------------------+
         |
         v
+-------------------+
| End               |
+-------------------+

```