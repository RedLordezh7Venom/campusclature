from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.rag_pipeline import ingest_pdf, get_qa_chain

app = FastAPI()

# Global QA Chain instance
qa_chain = None

# Request model for /ask endpoint
class QueryRequest(BaseModel):
    query: str

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    # Ingest the PDF
    ingest_pdf("temp.pdf")

    # Build the global QA chain with memory
    global qa_chain
    qa_chain = get_qa_chain()

    return {"message": "PDF uploaded and indexed successfully."}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    if qa_chain is None:
        return {"error": "No PDF has been uploaded yet."}

    # Ask the question via QA chain
    response = qa_chain.invoke({"question": request.query})

    # Print the memory summary in terminal (if memory exists)
    if hasattr(qa_chain, "memory") and hasattr(qa_chain.memory, "buffer"):
        print("\n🧠 Conversation Summary:\n", qa_chain.memory.buffer)

    return {"answer": response["answer"]}
