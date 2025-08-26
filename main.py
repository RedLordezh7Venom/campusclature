from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import ingest_pdf, get_qa_chain
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()

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

# Initial pipeline load
reload_pipeline()

# Watchdog event handler
class PDFHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path == PDF_PATH:
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

# Start the watcher thread when the FastAPI app starts
@app.on_event("startup")
async def startup_event():
    watcher_thread = threading.Thread(target=start_watcher, daemon=True)
    watcher_thread.start()

# Request model for /ask endpoint
class QueryRequest(BaseModel):
    query: str

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    if qa_chain is None:
        return {"error": "No PDF has been processed yet. Please ensure temp.pdf exists."}

    # Ask the question via QA chain
    response = qa_chain.invoke({"question": request.query})

    # Print the memory summary in terminal (if memory exists)
    if hasattr(qa_chain, "memory") and hasattr(qa_chain.memory, "buffer"):
        print("\n🧠 Conversation Summary:\n", qa_chain.memory.buffer)

    return {"answer": response["answer"]}
