"""
RAG Pipeline Module for Company Knowledge Chatbot
Built with LangChain, FAISS, and Google Gemini API.
"""
import os
import glob
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()


def get_api_key(provided_key: Optional[str] = None) -> str:
    """Resolve Gemini API key from parameter or environment variables."""
    key = provided_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key.strip() == "" or "your_" in key:
        raise ValueError(
            "Gemini API Key is missing! Please configure GEMINI_API_KEY in .env "
            "or enter it in the Streamlit sidebar."
        )
    return key.strip()


def load_company_documents(data_dir: str = "data") -> List[Document]:
    """
    Load all .txt and .xlsx documents from the local data directory.
    Uses LangChain TextLoader and UnstructuredExcelLoader (with robust fallback).
    """
    documents: List[Document] = []
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return documents

    # 1. Load .txt files using LangChain TextLoader
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(txt_file, encoding="utf-8")
            loaded = loader.load()
            documents.extend(loaded)
            print(f"[RAG] Successfully loaded {len(loaded)} doc(s) from {txt_file}")
        except Exception as e:
            print(f"[RAG Warning] Failed to load {txt_file} with TextLoader: {e}")

    # 2. Load .xlsx files using UnstructuredExcelLoader (with table fallback)
    xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    for xlsx_file in xlsx_files:
        loaded_excel = False
        # Try LangChain UnstructuredExcelLoader first
        try:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(xlsx_file, mode="elements")
            loaded = loader.load()
            if loaded:
                documents.extend(loaded)
                loaded_excel = True
                print(f"[RAG] Loaded {len(loaded)} elements from {xlsx_file} via UnstructuredExcelLoader")
        except Exception as e:
            print(f"[RAG Notice] UnstructuredExcelLoader fallback for {xlsx_file}: {e}")

        # Fallback to openpyxl/pandas if Unstructured encountered an issue
        if not loaded_excel:
            try:
                import pandas as pd
                excel_data = pd.ExcelFile(xlsx_file)
                for sheet_name in excel_data.sheet_names:
                    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
                    # Convert dataframe rows into clean readable markdown / structured documents
                    for idx, row in df.iterrows():
                        row_items = [f"**{col}**: {val}" for col, val in row.items() if pd.notna(val)]
                        content = f"Sheet: {sheet_name} | Entry #{idx + 1}\n" + "\n".join(row_items)
                        documents.append(
                            Document(
                                page_content=content,
                                metadata={"source": xlsx_file, "sheet": sheet_name, "row": idx + 1}
                            )
                        )
                print(f"[RAG] Successfully loaded {xlsx_file} via structured Excel reader.")
            except Exception as e2:
                print(f"[RAG Error] Unable to read Excel file {xlsx_file}: {e2}")

    return documents


def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split loaded documents into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vectorstore(documents: List[Document], api_key: Optional[str] = None) -> FAISS:
    """
    Create embeddings and index chunked documents into a FAISS vectorstore.
    """
    resolved_key = get_api_key(api_key)
    chunks = split_documents(documents)
    
    if not chunks:
        raise ValueError("No documents or chunks available to index in FAISS vectorstore!")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=resolved_key
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


STRICT_RAG_PROMPT = """You are Aura Technologies' helpful corporate assistant.
Your goal is to assist employees and clients with company policies, benefits, equipment allowances, and product specifications.

GUIDELINES:
1. GREETINGS & CONVERSATIONAL PLEASANTRIES:
   - For greetings (e.g., "hi", "hello", "good morning"), respond warmly and offer help with company policies or product specifications.
   - For goodbyes (e.g., "bye", "see you", "have a good day"), reply politely (e.g., "Goodbye! Have a great day ahead!").
   - For gratitude (e.g., "thank you", "thanks"), respond courteously (e.g., "You're welcome! Feel free to ask if you need anything else.").
   - For identity queries (e.g., "who are you?", "what can you do?"), briefly introduce yourself as the Aura Technologies Company Assistant.

2. FACTUAL COMPANY & PRODUCT QUESTIONS:
   - Answer using ONLY the retrieved context below. Consider the recent conversation history to understand follow-up questions and pronouns (e.g., "it", "that plan", "the first one").
   - Present numbers, policies, tiers, and pricing with clear, professional bullet points where applicable.
   - If the retrieved context does NOT contain enough information to answer a factual company question, respond politely:
     "I don't have information on that in Aura Technologies' documentation. Please reach out to HR or your team lead for clarification."
   - Do NOT assume, invent, or extrapolate policies, prices, or allowances not explicitly stated in the context.

Recent Conversation History:
---------------------
{history}
---------------------

Retrieved Context:
---------------------
{context}
---------------------

User Question: {question}

Answer:"""


REWRITE_PROMPT = """Given the recent conversation history and a user follow-up question, rephrase the follow-up question into a standalone, concise search query that captures all relevant entity names, product names, or policies.
If the question is already standalone or is a greeting/casual remark, return it unchanged. Do NOT answer the question.

Recent History:
{history}

Follow-up Question: {question}

Standalone Search Query:"""


def contextualize_query(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]],
    llm: ChatGoogleGenerativeAI
) -> str:
    """Uses LLM to rewrite ambiguous follow-up questions into standalone queries for precise retrieval."""
    if not chat_history:
        return query
    
    # Check if query is very short or contains conversational pronouns
    history_lines = []
    for msg in chat_history[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
            
    if not history_lines:
        return query

    try:
        rewrite_chain = ChatPromptTemplate.from_template(REWRITE_PROMPT) | llm | StrOutputParser()
        standalone = rewrite_chain.invoke({
            "history": "\n".join(history_lines),
            "question": query
        })
        cleaned = standalone.strip().strip('"')
        return cleaned if cleaned else query
    except Exception:
        return query


def stream_rag(
    query: str,
    vectorstore: FAISS,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
    k: int = 3,
    chat_history: Optional[List[Dict[str, Any]]] = None
):
    """
    Retrieves relevant documents with query reformulation and streams response tokens.
    Yields (stream_generator, source_documents).
    """
    resolved_key = get_api_key(api_key)
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=resolved_key
    )

    # 1. Conversational Query Rewriting for higher retrieval precision
    search_query = contextualize_query(query, chat_history, llm)

    # 2. Top-k similarity retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(search_query)

    # 3. Format context
    context_text = "\n\n".join(
        [f"[Source {i+1} | {doc.metadata.get('source', 'Unknown')}]:\n{doc.page_content}" 
         for i, doc in enumerate(retrieved_docs)]
    )

    # 4. History formatting
    history_lines = []
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if content:
                history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines) if history_lines else "No previous conversation."

    prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)
    chain = prompt | llm | StrOutputParser()

    stream_gen = chain.stream({
        "history": history_text,
        "context": context_text,
        "question": query
    })

    return stream_gen, retrieved_docs


def ask_rag(
    query: str,
    vectorstore: FAISS,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
    k: int = 3,
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Synchronous fallback wrapper for ask_rag using stream_rag generator.
    """
    stream_gen, retrieved_docs = stream_rag(
        query=query,
        vectorstore=vectorstore,
        api_key=api_key,
        model_name=model_name,
        k=k,
        chat_history=chat_history
    )
    answer = "".join(list(stream_gen))
    return {
        "answer": answer.strip(),
        "source_documents": retrieved_docs
    }

