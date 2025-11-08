from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from app.config import EMBEDDING_MODEL, FAISS_DIR
from app.pdf_loader import load_and_split_pdf
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import sys

# Configure logging with UTF-8 encoding to support emojis/Unicode
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

file_handler = RotatingFileHandler(
    'rag_pipeline.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'  # UTF-8 encoding for Unicode support
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

# Set stdout to UTF-8 mode on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

logger = logging.getLogger(__name__)

# --- Welida Custom Prompt ---
prompt_template = """
You are **Welida**, a study course generator. Your task is to generate course links based on the **user's query** using the provided **context only** , and you can chat normally with the user when the query is not course related.
The user may speak in **English, Hinglish, or any language** — respond accordingly.
---
### RULES:
* If the user talks casually or says anything unrelated to studying (e.g. "hi," "kya haal hai," "what's up"), reply normally as you would any other request, giving them info , conversationally.
* If the user expresses **any learning intent** (e.g. "vectors padhna hai," "physics chahiye," "numericals on motion") → generate a course link.
* **Always pick from the given context.** Never create or imagine a course.
* **Always reply with a course link.** If an exact match isn't available, give the **closest match**.
* **If multiple courses match**, rotate between them based on memory/history. Do not repeat the same course link in a row.
### Output format:
**Only return the course link from context when related to course queries. No title, no extra text, no emojis — just the link.**
Otherwise, reply conversationally, keeping it brief
---
### Golden Rule:
**Never hallucinate. Never skip. Never expose backend. Always reply.**
---
 
Chat History:  
{chat_history}
Context (Available Courses):  
{context}
User's Question:  
{question}
Welida's Response (rotate if repeated):**
"""

# Load environment variables
try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}", exc_info=True)
    raise

# Validate environment variables
def validate_env_vars():
    """
    Validate that required environment variables are present.
    """
    try:
        required_vars = ["OPENAI_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("All required environment variables are present")
        return True
        
    except Exception as e:
        logger.critical(f"Environment validation failed: {e}", exc_info=True)
        raise

# Validate on import
try:
    validate_env_vars()
except Exception as e:
    logger.critical(f"Failed to validate environment variables: {e}", exc_info=True)
    # Don't raise here, let the calling code handle it

# Create prompt template
try:
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template
    )
    logger.info("Prompt template created successfully")
except Exception as e:
    logger.error(f"Error creating prompt template: {e}", exc_info=True)
    raise

# --- Step 1: Load + Embed PDF ---
def ingest_pdf(pdf_path: str) -> bool:
    """
    Ingest a PDF file, split it into chunks, and create a FAISS vector store.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF is empty or invalid
        Exception: For other errors during ingestion
    """
    try:
        logger.info(f"Starting PDF ingestion for: {pdf_path}")
        
        # Validate PDF path
        if not pdf_path:
            raise ValueError("PDF path cannot be empty")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")
        
        # Load and split PDF
        logger.info("Loading and splitting PDF...")
        docs = load_and_split_pdf(pdf_path)
        
        if not docs:
            raise ValueError("No documents extracted from PDF")
        
        logger.info(f"Successfully extracted {len(docs)} document chunks")
        
        # Validate embedding model
        if EMBEDDING_MODEL is None:
            raise ValueError("EMBEDDING_MODEL is not configured")
        
        logger.info(f"Creating FAISS vector store with {len(docs)} documents")
        
        # Create vector store
        vectordb = FAISS.from_documents(
            documents=docs,
            embedding=EMBEDDING_MODEL
        )
        
        logger.info("Vector store created successfully")
        
        # Ensure FAISS directory exists
        os.makedirs(FAISS_DIR, exist_ok=True)
        logger.info(f"Ensured FAISS directory exists: {FAISS_DIR}")
        
        # Save vector store
        logger.info(f"Saving vector store to: {FAISS_DIR}")
        vectordb.save_local(FAISS_DIR)
        
        logger.info("PDF ingestion completed successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}", exc_info=True)
        raise
    except ValueError as e:
        logger.error(f"Invalid PDF or configuration: {e}", exc_info=True)
        raise
    except PermissionError as e:
        logger.error(f"Permission denied: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF ingestion: {e}", exc_info=True)
        raise

# --- Step 2: RAG chain using OpenAI GPT-4o + Summary Memory ---
def get_qa_chain() -> Optional[ConversationalRetrievalChain]:
    """
    Create and return a ConversationalRetrievalChain with FAISS retriever and memory.
    
    Returns:
        ConversationalRetrievalChain: The configured QA chain
        
    Raises:
        FileNotFoundError: If FAISS index doesn't exist
        ValueError: If configuration is invalid
        Exception: For other errors during chain creation
    """
    try:
        logger.info("Initializing QA chain...")
        
        # Validate FAISS directory exists
        if not os.path.exists(FAISS_DIR):
            raise FileNotFoundError(
                f"FAISS directory not found: {FAISS_DIR}. "
                "Please ingest a PDF first."
            )
        
        # Check for FAISS index file
        index_file = os.path.join(FAISS_DIR, "index.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FAISS index file not found: {index_file}. "
                "Please ingest a PDF first."
            )
        
        logger.info(f"Loading FAISS vector store from: {FAISS_DIR}")
        
        # Load FAISS vector store
        try:
            vectordb = FAISS.load_local(
                FAISS_DIR,
                EMBEDDING_MODEL,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}", exc_info=True)
            raise ValueError(f"Failed to load FAISS index: {e}")
        
        # Configure retriever with MMR search
        logger.info("Configuring retriever with MMR search strategy")
        try:
            retriever = vectordb.as_retriever(
                search_type="mmr",  # Maximal Marginal Relevance for diversity
                search_kwargs={
                    "k": 10,  # Get more results
                    "fetch_k": 20,  # Fetch more before filtering
                    "lambda_mult": 0.7
                }
            )
            logger.info("Retriever configured successfully")
        except Exception as e:
            logger.error(f"Error configuring retriever: {e}", exc_info=True)
            raise
        
        # Debug function for retriever (can be used for testing)
        def debug_retriever(query: str):
            """
            Debug function to inspect retriever results.
            
            Args:
                query: The search query
                
            Returns:
                List of retrieved documents
            """
            try:
                logger.debug(f"Debug retriever called with query: {query}")
                docs = retriever.get_relevant_documents(query)
                logger.debug(f"Retrieved {len(docs)} documents")
                
                for i, doc in enumerate(docs):
                    logger.debug(f"Doc {i}: {doc.page_content[:100]}...")
                
                return docs
            except Exception as e:
                logger.error(f"Error in debug_retriever: {e}", exc_info=True)
                return []
        
        # Initialize LLM
        logger.info("Initializing ChatOpenAI LLM")
        try:
            api_key = os.getenv("OPENAI_KEY")
            if not api_key:
                raise ValueError("OPENAI_KEY not found in environment variables")
            
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo"
                max_tokens=512,
                temperature=0.4,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}", exc_info=True)
            raise
        
        # Initialize conversation memory
        logger.info("Initializing conversation memory")
        try:
            memory = ConversationSummaryMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True
            )
            logger.info("Conversation memory initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing memory: {e}", exc_info=True)
            raise
        
        # Create conversational retrieval chain
        logger.info("Creating ConversationalRetrievalChain")
        try:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt},
            )
            logger.info("QA chain created successfully")
            return qa_chain
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}", exc_info=True)
            raise
        
    except FileNotFoundError as e:
        logger.error(f"FAISS index not found: {e}", exc_info=True)
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating QA chain: {e}", exc_info=True)
        raise

# Health check function
def check_pipeline_health() -> dict:
    """
    Check the health status of the RAG pipeline.
    
    Returns:
        dict: Health status information
    """
    try:
        health_status = {
            "faiss_dir_exists": os.path.exists(FAISS_DIR),
            "faiss_index_exists": os.path.exists(os.path.join(FAISS_DIR, "index.faiss")),
            "embedding_model_configured": EMBEDDING_MODEL is not None,
            "openai_key_configured": bool(os.getenv("OPENAI_KEY")),
        }
        
        logger.info(f"Pipeline health check: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"Error during health check: {e}", exc_info=True)
        return {"error": str(e)}