from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.rag_pipeline import ingest_pdf, get_qa_chain
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from contextlib import asynccontextmanager
import re

# Global QA Chain instance
qa_chain = None
PDF_PATH = "temp.pdf"

# Function to initialize/reload the RAG pipeline
def reload_pipeline():
    global qa_chain
    print(f"🔄 Reloading RAG pipeline from {PDF_PATH}...")
    if os.path.exists(PDF_PATH):
        ingest_pdf(PDF_PATH)
        qa_chain = get_qa_chain()
        print("✅ RAG pipeline reloaded successfully.")
    else:
        print(f"⚠️ {PDF_PATH} not found. RAG pipeline not loaded.")
        qa_chain = None

# Watchdog event handler
class PDFHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(PDF_PATH):
            print(f"Detected change in {event.src_path}. Reloading pipeline...")
            reload_pipeline()

# Start watchdog observer in a separate thread
def start_watcher():
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    reload_pipeline()
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()
    
    yield  # App runs here

    # Shutdown (optional cleanup can go here)

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan, root_path="/api")

# Request model for /ask endpoint
class QueryRequest(BaseModel):
    query: str
@app.get("/")
async def root():
    return {"message": "RAG pipeline is running. See /docs for API."}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain, pdf_last_modified

    try:
        with open(PDF_PATH, "wb") as f:
            content = await file.read()
            f.write(content)

        # Update last modified time after saving the file
        pdf_last_modified = os.path.getmtime(PDF_PATH)

        # Reload the RAG pipeline immediately
        qa_chain = load_qa_pipeline()

        print(f"📄 {file.filename} uploaded successfully and saved as {PDF_PATH}.")
        print("🔄 RAG pipeline reloaded after upload.")
        return {"message": f"File '{file.filename}' uploaded successfully. RAG pipeline reloaded."}
    except Exception as e:
        return {"error": f"Could not upload file: {e}"}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    if qa_chain is None:
        return {"error": "No PDF has been processed yet. Please ensure temp.pdf exists."}

    # Ask the question via QA chain
    response = qa_chain.invoke({"question": request.query})

    # Print the memory summary in terminal (if memory exists)
    if hasattr(qa_chain, "memory") and hasattr(qa_chain.memory, "buffer"):
        print("\n🧠 Conversation Summary:\n", qa_chain.memory.buffer)
    pattern = r"^https://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[\w\-._~:/?#[\]@!$&'()*+,;=]*)?$"
    is_match = bool(re.match(pattern,response["answer"]))
    if is_match:
        return {"answer": None,"course_link":response["answer"]}
    else:
        return {"answer": response["answer"],"course_link":None}

    

