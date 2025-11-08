from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag_pipeline import ingest_pdf, get_qa_chain
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from contextlib import asynccontextmanager
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Console handler
        logging.StreamHandler(),
        # File handler with rotation
        RotatingFileHandler(
            'rag_pipeline.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)

logger = logging.getLogger(__name__)

# Global QA Chain instance
qa_chain = None
PDF_PATH = "temp.pdf"

# Function to initialize/reload the RAG pipeline
def reload_pipeline():
    """
    Initialize or reload the RAG pipeline from the PDF file.
    Handles exceptions and logs the process.
    """
    global qa_chain
    try:
        logger.info(f"Attempting to reload RAG pipeline from {PDF_PATH}")
        
        if not os.path.exists(PDF_PATH):
            logger.warning(f"{PDF_PATH} not found. RAG pipeline not loaded.")
            qa_chain = None
            return False
        
        # Check if file is readable and not empty
        file_size = os.path.getsize(PDF_PATH)
        if file_size == 0:
            logger.error(f"{PDF_PATH} is empty. Cannot load RAG pipeline.")
            qa_chain = None
            return False
        
        logger.info(f"PDF file size: {file_size} bytes")
        
        # Ingest PDF
        logger.info("Starting PDF ingestion...")
        ingest_pdf(PDF_PATH)
        logger.info("PDF ingestion completed successfully")
        
        # Get QA chain
        logger.info("Initializing QA chain...")
        qa_chain = get_qa_chain()
        logger.info("QA chain initialized successfully")
        
        logger.info("✅ RAG pipeline reloaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}", exc_info=True)
        qa_chain = None
        return False
    except PermissionError as e:
        logger.error(f"Permission denied accessing {PDF_PATH}: {e}", exc_info=True)
        qa_chain = None
        return False
    except Exception as e:
        logger.error(f"Unexpected error during pipeline reload: {e}", exc_info=True)
        qa_chain = None
        return False

# Watchdog event handler
class PDFHandler(FileSystemEventHandler):
    """
    File system event handler for monitoring PDF changes.
    """
    def on_modified(self, event):
        try:
            if event.src_path.endswith(PDF_PATH):
                logger.info(f"Detected change in {event.src_path}")
                reload_pipeline()
        except Exception as e:
            logger.error(f"Error handling file modification event: {e}", exc_info=True)

# Start watchdog observer in a separate thread
def start_watcher():
    """
    Start the file system watcher in a separate thread.
    Monitors for changes to the PDF file.
    """
    try:
        logger.info("Starting file system watcher...")
        event_handler = PDFHandler()
        observer = Observer()
        observer.schedule(event_handler, path='.', recursive=False)
        observer.start()
        logger.info("File system watcher started successfully")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping watcher...")
            observer.stop()
        
        observer.join()
        logger.info("File system watcher stopped")
        
    except Exception as e:
        logger.error(f"Error in file system watcher: {e}", exc_info=True)

# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    try:
        logger.info("="*60)
        logger.info("Starting RAG Pipeline Application")
        logger.info(f"Startup time: {datetime.now()}")
        logger.info("="*60)
        
        reload_pipeline()
        
        logger.info("Starting watcher thread...")
        watcher_thread = threading.Thread(target=start_watcher, daemon=True)
        watcher_thread.start()
        logger.info("Watcher thread started successfully")
        
    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)
        raise
    
    yield  # App runs here
    
    # Shutdown
    try:
        logger.info("="*60)
        logger.info("Shutting down RAG Pipeline Application")
        logger.info(f"Shutdown time: {datetime.now()}")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}", exc_info=True)

# Create FastAPI app with lifespan
app = FastAPI(
    lifespan=lifespan,
    root_path="/api",
    title="RAG Pipeline API",
    description="API for RAG-based question answering with PDF documents",
    version="1.0.0"
)

# Request model for /ask endpoint
class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the main topic of the document?"
            }
        }

@app.get("/")
async def root():
    """
    Root endpoint providing API status.
    """
    try:
        logger.info("Root endpoint accessed")
        pipeline_status = "loaded" if qa_chain is not None else "not loaded"
        return {
            "message": "RAG pipeline is running. See /docs for API.",
            "pipeline_status": pipeline_status,
            "pdf_path": PDF_PATH,
            "pdf_exists": os.path.exists(PDF_PATH)
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and reload the RAG pipeline.
    """
    global qa_chain
    
    try:
        logger.info(f"Received file upload request: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # Validate file size (optional, adjust as needed)
        content = await file.read()
        file_size = len(content)
        logger.info(f"File size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Uploaded file is empty")
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            logger.warning(f"File too large: {file_size} bytes")
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 50MB limit"
            )
        
        # Write file to disk
        logger.info(f"Writing file to {PDF_PATH}")
        with open(PDF_PATH, "wb") as f:
            f.write(content)
        logger.info(f"File written successfully: {file.filename}")
        
        # Reload the RAG pipeline immediately after uploading the PDF
        logger.info("Reloading RAG pipeline after upload...")
        success = reload_pipeline()
        
        if not success:
            logger.error("Failed to reload RAG pipeline after upload")
            raise HTTPException(
                status_code=500,
                detail="File uploaded but pipeline reload failed. Check logs."
            )
        
        logger.info(f"✅ Upload complete and pipeline reloaded for {file.filename}")
        return {
            "message": f"File '{file.filename}' uploaded successfully. RAG pipeline reloaded.",
            "file_size": file_size,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not upload file: {str(e)}"
        )

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    """
    Ask a question to the RAG pipeline.
    """
    try:
        logger.info(f"Received question: {request.query}")
        
        # Validate query
        if not request.query or not request.query.strip():
            logger.warning("Empty query received")
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if qa_chain is None:
            logger.error("QA chain not initialized when question was asked")
            raise HTTPException(
                status_code=503,
                detail="No PDF has been processed yet. Please upload a PDF first."
            )
        
        # Ask the question via QA chain
        logger.info("Invoking QA chain...")
        response = qa_chain.invoke({"question": request.query})
        logger.info("QA chain invoked successfully")
        
        # Print the memory summary in terminal (if memory exists)
        if hasattr(qa_chain, "memory") and hasattr(qa_chain.memory, "buffer"):
            logger.info(f"Conversation Summary: {qa_chain.memory.buffer}")
        
        # Check if response is a URL
        pattern = r"^https://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[\w\-._~:/?#[\]@!$&'()*+,;=]*)?$"
        is_match = bool(re.match(pattern, response["answer"]))
        
        if is_match:
            logger.info(f"Response is a course link: {response['answer']}")
            return {
                "answer": None,
                "course_link": response["answer"],
                "status": "success"
            }
        else:
            logger.info("Response is a text answer")
            return {
                "answer": response["answer"],
                "course_link": None,
                "status": "success"
            }
        
    except HTTPException:
        raise
    except KeyError as e:
        logger.error(f"Missing expected key in response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Invalid response format from QA chain"
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    try:
        return {
            "status": "healthy",
            "pipeline_loaded": qa_chain is not None,
            "pdf_exists": os.path.exists(PDF_PATH),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")