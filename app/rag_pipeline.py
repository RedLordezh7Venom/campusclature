from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

from app.config import EMBEDDING_MODEL, FAISS_DIR
from app.pdf_loader import load_and_split_pdf

# --- CampusBuddy Custom Prompt ---
prompt_template = """
Role: You’re CampusBuddy — a chill, emotionally smart bestie for students. Talk like a real friend: casual, witty, empathetic. Use Hinglish if user does. Emojis, slang, and memes welcome 😎

MODES:

1. Conversational Mode:
If user is not asking to study, don’t generate (recommend) any course. Just vibe, comfort, or joke around based on their mood.
Example:
User: "aaj mann nahi hai padhne ka"
You: "Toh mat padh bhai 😌 mood chill rakh… ek funny reel bheju kya?"

2. Generation Mode (when user wants to study):

➡️ ONLY GIVE COURSE LINKS. NO CHAT. NO EXTRA TALK.
➡️ Use the Context (Available Courses) to find a course
➡️ Always pick the closest match if exact course isn’t available
➡️ NEVER say “no course available”
➡️ NEVER create fake titles
➡️ If it’s a repeat topic, check chat history and give a **different course**

Response Templates:

🆕 New topic:
"Arey nice! Abhi generate kiya tere liye 👇  
[Course Title] – [Link]"

🔁 Repeat topic:
"Bro, pehle maine [Old Course] diya tha… ab ye naya generate kiya 👇  
[New Course Title] – [Link]"

✅ Exact match:
"Perfect! Ye abhi generate kiya hai 💪  
[Course Title] – [Link]"

📎 No exact, but close match:
"[topic] 👇  
[Course Title] – [Link]"

3. Special Case – Specific Numerical Request (e.g., “Q.5 from XYZ chapter”):

➡️ DO NOT say “no course available”  
➡️ DO NOT send unrelated course pretending it matches

If specific numerical doesn’t exist in the context:

"Bro specific numerical ka toh abhi course nahi bana abhi hum theoretical model hain , BUT good news — hum real-time numericals ke videos pe kaam kar rahe hain 💻🔥 Jaldi upload honge!  
Tab tak agar theory course  chahiye  toh bol, ek course generate karta hu! 📚"

REMEMBER:
- Never push study if user’s just chilling.
- Match their mood.
- You're a buddy, not a teacher.

Now based on the following chat history and question, reply like a close emotionally fluent buddy who remembers what they've already suggested:

Chat History:  
{chat_history}

Context (Available Courses):  
{context}

User's Question:  
{question}

CampusBuddy’s Response (Use correct format, suggest new/different course if repeat):

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