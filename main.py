
# Import required libraries
import chromadb
import fitz  # PyMuPDF for PDF processing
import uvicorn
from fastapi import FastAPI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()


# Global variables for embeddings and database
MODEL_NAME = "all-MiniLM-L6-v2"  # Name of the model to be used for generating embeddings
EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)  # Load the embedding model
VECTOR_DB = chromadb.Client()  # Initialize the ChromaDB client
COLLECTION = VECTOR_DB.get_or_create_collection("pdf_knowledge_base")  # Get or create a collection in ChromaDB

# OpenAI API configuration
OPENAI_API_KEY = "<your_openai_api_key>"  # Your OpenAI API key
OPENAI_API_TYPE = "azure"  # The type of API service being used, e.g., Azure
OPENAI_BASE_URL = "<your_openai_base_url>"  # The base URL for the OpenAI API
OPENAI_BASE_VERSION = "<your_openai_api_version>"  # The version of the OpenAI API
OPEN_AI_MODEL = "gpt-4o"  # The OpenAI model to be used


class QueryRequest(BaseModel):
    # Define a Pydantic model for the request body
    query: str


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def preprocess_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def index_pdf_text(pdf_files):
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = preprocess_text(text)
        embeddings = EMBEDDING_MODEL.encode(chunks)

        ids = [f"{pdf_path}_chunk_{i}" for i in range(len(chunks))]

        COLLECTION.add(
            documents=chunks,
            metadatas=[{"source": pdf_path}] * len(chunks),
            embeddings=[embedding.tolist() for embedding in embeddings],
            ids=ids
        )
    print("Index completed...")


def query_knowledge_base(query, top_k=3):

    client = AzureOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_BASE_URL,
        api_version=OPENAI_BASE_VERSION,
    )

    results = COLLECTION.query(query_texts=[query], n_results=top_k)
    context = " ".join(result[0] for result in results["documents"])
    print(context)
    response = client.chat.completions.create(
        model=OPEN_AI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ]
    )
    return response.choices[0].message.content


# Step 5: API Endpoint
@app.post("/query")
def ask_agent(request: QueryRequest):
    try:
        response = query_knowledge_base(request.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}


# Main function to run the script
if __name__ == "__main__":
    # List of PDF files to process
    pdf_files = ["sample_1.pdf", "sample_2.pdf", "sample_3.pdf", "sample_4.pdf", "sample_5.pdf"]
    print("Starting indexing process...")
    index_pdf_text(pdf_files)
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

