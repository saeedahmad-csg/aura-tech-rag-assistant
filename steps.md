# Project 2 — RAG-Based Custom Chatbot with Streamlit

A Step-by-Step Guide to setting up, running, and presenting your company knowledge RAG chatbot using Streamlit, LangChain, FAISS, and Google Gemini API.

---

## 📋 Architecture Overview
- **Data Ingestion**: Loads local company policies (`data/company_info.txt`) and product catalog (`data/products.xlsx`) using LangChain loaders (`TextLoader` + `UnstructuredExcelLoader`).
- **Chunking & Indexing**: Chunks text with `RecursiveCharacterTextSplitter` (chunk size ~500, overlap ~50) and generates embeddings with Google Generative AI (`models/text-embedding-004`).
- **Vector Search**: Indexes vectors in memory using FAISS (`faiss-cpu`) with top-3 similarity search retrieval (`k=3`).
- **Generation**: Uses Google Gemini (`gemini-2.0-flash-lite` or `gemini-2.5-flash-lite`) with a strict grounding prompt. If the answer is not in the context, it strictly answers *"I don't know"*.
- **UI**: Streamlit chat interface (`st.chat_input`, `st.chat_message`) with `@st.cache_resource` for one-time startup loading and conversation history.

---

## 🚀 Step 1: Environment Setup & Installation

1. Open your terminal in the project directory:
   ```powershell
   cd "d:\Projects\17- RAG Remake"
   ```

2. (Optional but recommended) Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install required libraries:
   ```powershell
   pip install -r requirements.txt
   ```

4. Configure your Gemini API key:
   - Open `.env` and set:
     ```env
     GEMINI_API_KEY=your_actual_gemini_api_key
     ```
   - *Note: You can also enter or override your Gemini API key directly in the Streamlit app sidebar.*

---

## 📁 Step 2: Preparing Company Knowledge Data

The data lives in the `data/` folder:
- `data/company_info.txt`: Contains official corporate policies, remote work guidelines, equipment allowance, and office hours for *Aura Technologies*.
- `data/products.xlsx`: Contains an Excel table with software subscription tiers, pricing, storage allotments, and feature sets.

To generate or regenerate the sample Excel file if needed:
```powershell
python create_data.py
```

---

## 🧠 Step 3: RAG Core Pipeline (`rag.py`)

- `load_company_documents()`: Reads both `.txt` and `.xlsx` files and aggregates them into LangChain `Document` objects.
- `build_vectorstore(documents)`: Splits documents into chunks and builds a FAISS index using `GoogleGenerativeAIEmbeddings`.
- `get_rag_chain(vectorstore)`: Creates the retrieval chain with `ChatGoogleGenerativeAI` and the grounded prompt.

---

## 💻 Step 4: Streamlit Chat Application (`app.py`)

Launch the web application:
```powershell
streamlit run app.py
```

Features included:
- **One-time startup caching**: Vectorstore and documents load once via `@st.cache_resource`. Subsequent questions retrieve instantly without reloading files.
- **Interactive chat**: Real-time conversation with session history.
- **Source inspection**: Expandable drawer under each assistant response showing the exact 3 source chunks retrieved.
- **Sidebar controls**:
  - API key status and manual override.
  - Model selector (e.g. `gemini-2.0-flash-lite`, `gemini-2.5-flash-lite`).
  - Document index summary.
  - "Clear Chat" button.

---

## 🧪 Step 5: Manual Verification Checklist

Test the chatbot with these questions to verify all requirements:

| Test Case | Question | Expected Behavior |
| :--- | :--- | :--- |
| **1. Company Info (Text)** | *"What is the annual equipment reimbursement allowance?"* | Answers accurately from `company_info.txt` ($1,200 annually). |
| **2. Company Policy (Text)** | *"What are the official core support hours?"* | Answers accurately (9:00 AM – 6:00 PM EST). |
| **3. Excel Data (Pricing)** | *"How much does the Enterprise plan cost per month?"* | Answers accurately from `products.xlsx` ($499/month). |
| **4. Excel Data (Features)** | *"Which plan includes dedicated 24/7 phone support and 5 TB storage?"* | Identifies the Enterprise / Custom plan. |
| **5. Out of Domain ("I don't know")** | *"What is the recipe for chocolate chip cookies?"* | Strictly answers: **"I don't know."** |
| **6. World Knowledge ("I don't know")** | *"Who won the FIFA World Cup in 2022?"* | Strictly answers: **"I don't know."** |

---

## 📢 Step 6: LinkedIn Submission Kit

### Screenshot Recommendations:
1. One screenshot showing a successful answer retrieved from `company_info.txt`.
2. One screenshot showing an answer retrieved from `products.xlsx` with the "Retrieved Sources" expander open.
3. One screenshot demonstrating the fallback guardrail responding *"I don't know"* for an external question.

### Post Template:
```text
🚀 Excited to share my latest AI project: A RAG-Based Custom Company Chatbot built with Streamlit, LangChain, FAISS, and Google Gemini!

🏢 The Challenge:
Employees frequently spend time searching through internal documents and spreadsheets to find company policies and product information. 

💡 The Solution:
I built an intelligent enterprise assistant that loads company data (.txt policies and .xlsx product catalogs) at startup, chunks and indexes the content into a FAISS vector store, and answers employee queries strictly using retrieved context.

Key Features:
✅ Multi-format ingestion with LangChain TextLoader & Excel loaders
✅ High-performance vector search with FAISS
✅ Grounded responses powered by Google Gemini (Flash-Lite)
✅ Strict hallucination guardrail: Answers "I don't know" when information is out-of-scope
✅ Responsive Streamlit chat interface with full conversation history & source transparency

#AI #GenerativeAI #RAG #LangChain #Streamlit #GoogleGemini #Python #MachineLearning
```

