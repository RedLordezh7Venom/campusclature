from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

from app.config import EMBEDDING_MODEL, FAISS_DIR
from app.pdf_loader import load_and_split_pdf

# --- Welida Custom Prompt ---
prompt_template = """


You are **Welida**, a study course generator. Your task is to generate course links based on the **user’s query** using the provided **context only** , and you can chat normally with the user when the query is not course related.

The user may speak in **English, Hinglish, or any language** — respond accordingly.

---

### RULES:

* If the user talks casually or says anything unrelated to studying (e.g. "hi," "kya haal hai," "what's up"), reply normally as you would any other request, giving them info , conversationally.

* If the user expresses **any learning intent** (e.g. “vectors padhna hai,” “physics chahiye,” “numericals on motion”) → generate a course link.

* **Always pick from the given context.** Never create or imagine a course.

* **Always reply with a course link.** If an exact match isn’t available, give the **closest match**.

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

Welida’s Response (rotate if repeated):**

"""
load_dotenv()
api_key = os.getenv("GROQ_KEY")
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# --- Step 1: Load + Embed PDF ---
def ingest_pdf(pdf_path: str):
    docs = load_and_split_pdf(pdf_path)
    vectordb = FAISS.from_documents(
        documents=docs,
        embedding=EMBEDDING_MODEL
    )
    vectordb.save_local(FAISS_DIR)

# --- Step 2: RAG chain using OpenRouter GPT-4o + Summary Memory ---

def get_qa_chain():
    vectordb = FAISS.load_local(FAISS_DIR, EMBEDDING_MODEL, allow_dangerous_deserialization=True)
    # Try different search strategies
    retriever = vectordb.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={
            "k": 10,  # Get more results
            "fetch_k": 20,  # Fetch more before filtering
            "lambda_mult": 0.7
        }
    )
    def debug_retriever(query):
        docs = retriever.get_relevant_documents(query)
        print(f"Query: {query}")
        for i, doc in enumerate(docs):
            print(f"Doc {i}: {doc.page_content[:100]}...")
        return docs
    llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_KEY"),
    model="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo"
    max_tokens=512,
    temperature=0.4,
)

    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )