"""
Streamlit Web Application for Company Knowledge RAG Chatbot
Uses LangChain, FAISS, and Google Gemini API.
"""
import os
import streamlit as st
from dotenv import load_dotenv

import rag

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Aura Tech — Company Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, modern enterprise UI styling (Clean Light Theme)
st.markdown("""
<style>
    /* Ensure Streamlit sidebar collapsed control chevron is always accessible */
    [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Hide the Deploy button & top toolbar clutter */
    .stDeployButton, 
    [data-testid="stDeployButton"],
    [data-testid="stToolbarActions"],
    header [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Base Container Max Width & Clean Light Background */
    body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }

    /* Hero Banner Card */
    .hero-banner {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.5;
        margin: 0;
    }

    /* Sidebar Branding */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    .sidebar-brand-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .brand-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #0F172A;
        margin-top: 8px;
        margin-bottom: 2px;
    }

    /* Quick question chips */
    .quick-section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    div[data-testid="column"] button {
        border-radius: 10px !important;
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #1E293B !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease-in-out !important;
        min-height: 44px !important;
    }
    div[data-testid="column"] button:hover {
        border-color: #3B82F6 !important;
        color: #1D4ED8 !important;
        background-color: #F8FAFC !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.1) !important;
    }

    /* High-Contrast Chat Messages Typography (LinkedIn Ready) */
    .stChatMessage, [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        padding: 0.85rem 1.15rem !important;
        margin-bottom: 0.85rem !important;
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    }
    .stChatMessage p, 
    .stChatMessage span, 
    .stChatMessage div, 
    .stChatMessage li, 
    .stChatMessage strong,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color: #0F172A !important;
        font-size: 0.96rem !important;
        line-height: 1.6 !important;
    }
    .stChatMessage strong {
        font-weight: 700 !important;
        color: #020617 !important;
    }

    /* Crisp High-Contrast Source citations styling */
    .source-box {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 4px solid #2563EB !important;
        padding: 12px 16px !important;
        margin-top: 8px !important;
        border-radius: 8px !important;
        font-size: 0.90rem !important;
        color: #0F172A !important;
        line-height: 1.6 !important;
    }
    .source-box * {
        color: #0F172A !important;
    }

    /* Expander Header High Contrast */
    [data-testid="stExpander"] {
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #0F172A !important;
    }

    /* ======================================================== */
    /* Mobile & Android Responsive Breakpoints (<= 768px)       */
    /* ======================================================== */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
            padding-bottom: 5rem !important;
            max-width: 100% !important;
        }
        .hero-banner {
            padding: 16px 18px !important;
            border-radius: 12px !important;
            margin-bottom: 16px !important;
        }
        .hero-title {
            font-size: 1.35rem !important;
            gap: 8px !important;
        }
        .hero-subtitle {
            font-size: 0.85rem !important;
            line-height: 1.4 !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        div[data-testid="column"] {
            flex: 1 1 45% !important;
            min-width: 140px !important;
        }
        div[data-testid="column"] button {
            font-size: 0.8rem !important;
            padding: 8px 10px !important;
            width: 100% !important;
        }
        .stChatMessage {
            padding: 0.5rem 0.75rem !important;
            font-size: 0.92rem !important;
        }
        [data-testid="stBottom"] {
            padding-bottom: env(safe-area-inset-bottom, 8px) !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-card">
        <img src="https://img.icons8.com/clouds/200/company.png" width="80" style="margin-bottom: 4px;"/>
        <div class="brand-title">Aura Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_model = "gemini-3.5-flash-lite"
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

    st.markdown("""
    <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px; line-height: 1.5;">
        Answers are strictly grounded in official documents:
        <br/>• 📄 <code>company_info.txt</code>
        <br/>• 📊 <code>products.xlsx</code>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<div style='margin-top: 30px; font-size: 0.78rem; color: #94A3B8; text-align: center;'>Powered by Gemini 3.5 Flash Lite • FAISS</div>", unsafe_allow_html=True)


def get_data_modification_fingerprint(data_dir: str = "data") -> float:
    """Calculates latest modification timestamp of files in data/ for automatic sync."""
    if not os.path.exists(data_dir):
        return 0.0
    latest_mtime = 0.0
    for root, _, files in os.walk(data_dir):
        for f in files:
            fp = os.path.join(root, f)
            try:
                latest_mtime = max(latest_mtime, os.path.getmtime(fp))
            except OSError:
                pass
    return latest_mtime


# Cached vectorstore creation - Auto-syncs whenever files in data/ change
@st.cache_resource(show_spinner="Indexing company knowledge files at startup...")
def initialize_knowledge_base(api_key: str, data_fingerprint: float):
    """
    Loads company documents from data/ and indexes them into FAISS.
    Automatically re-indexes if data_fingerprint changes (files modified/added).
    """
    if not api_key or api_key.strip() == "":
        return None, 0, "No API key provided."
    
    try:
        docs = rag.load_company_documents("data")
        if not docs:
            return None, 0, "No documents found in data/ directory."
        
        vectorstore = rag.build_vectorstore(docs, api_key=api_key)
        return vectorstore, len(docs), None
    except Exception as e:
        return None, 0, str(e)


# App Header Hero Card
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span>🏢</span> Aura Technologies Assistant
    </div>
    <div class="hero-subtitle">
        Ask questions about company policies, equipment allowances, or product tiers & pricing. 
        Answers are strictly grounded in verified internal documentation.
    </div>
</div>
""", unsafe_allow_html=True)

# Validate API Key & Load Knowledge Base
active_key = env_key.strip()

if not active_key:
    st.warning("⚠️ Please provide a Gemini API Key in `.env` to start chatting.", icon="🔑")
    st.stop()

# Initialize or retrieve cached VectorStore with auto-sync fingerprint
data_fingerprint = get_data_modification_fingerprint("data")
vectorstore, doc_count, init_error = initialize_knowledge_base(active_key, data_fingerprint)

if init_error:
    st.error(f"❌ Error initializing Knowledge Base: {init_error}")
    st.stop()

# Quick-Prompt Starter Chips
st.markdown('<div class="quick-section-title">💡 Frequently Asked Questions</div>', unsafe_allow_html=True)
chip_cols = st.columns(4)
clicked_prompt = None

starter_queries = [
    ("💼 Home Office", "What is the equipment & home office allowance policy?"),
    ("🌴 Leave Policy", "What is the annual leave and sick leave policy?"),
    ("💳 Pricing Tiers", "Compare the features and pricing of Pro and Enterprise plans."),
    ("🏥 Health Benefits", "What health and wellness benefits are provided?")
]

for col, (label, query_text) in zip(chip_cols, starter_queries):
    with col:
        if st.button(label, use_container_width=True):
            clicked_prompt = query_text

# Initialize Chat Message History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am Aura's corporate assistant. How can I help you today with company policies or product specifications?",
            "sources": []
        }
    ]

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display retrieved sources expander if available
        if msg.get("sources"):
            with st.expander("🔍 View Retrieved Sources (Top 3 Chunks)"):
                for idx, doc in enumerate(msg["sources"]):
                    src = doc.metadata.get("source", "Internal Document")
                    st.markdown(f"**Source {idx + 1}:** `{src}`")
                    st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)

# User Chat Input or Quick Chip Click
text_input = st.chat_input("Ask a question about company policies or products...")
effective_prompt = clicked_prompt or text_input

if effective_prompt:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": effective_prompt})
    with st.chat_message("user"):
        st.markdown(effective_prompt)

    # Generate Assistant Response (Real-time Streaming with Thinking Indicator)
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        with thinking_placeholder:
            st.markdown("""
            <div style="display: inline-flex; align-items: center; gap: 8px; color: #2563EB; font-weight: 500; font-size: 0.92rem; padding: 4px 8px;">
                <span>🧠</span> <em>Thinking & searching knowledge base...</em>
            </div>
            """, unsafe_allow_html=True)

        try:
            stream_gen, source_docs = rag.stream_rag(
                query=effective_prompt,
                vectorstore=vectorstore,
                api_key=active_key,
                model_name=selected_model,
                k=3,
                chat_history=st.session_state.messages[:-1]
            )

            # Define a generator that clears the thinking indicator on the very first token
            def stream_with_thinking_cleared(generator, placeholder):
                cleared = False
                for token in generator:
                    if not cleared:
                        placeholder.empty()
                        cleared = True
                    yield token
                if not cleared:
                    placeholder.empty()

            # Stream response in real-time
            answer_text = st.write_stream(stream_with_thinking_cleared(stream_gen, thinking_placeholder))

            # Render sources only if the query was factual (not a greeting/pleasantry)
            is_chit_chat = any(w in effective_prompt.lower().strip() for w in ["hi", "hello", "hey", "bye", "goodbye", "thanks", "thank you"])
            show_sources = source_docs and not (is_chit_chat and len(effective_prompt.split()) <= 3)

            if show_sources:
                with st.expander("🔍 View Retrieved Sources (Top 3 Chunks)"):
                    for idx, doc in enumerate(source_docs):
                        src = doc.metadata.get("source", "Internal Document")
                        st.markdown(f"**Source {idx + 1}:** `{src}`")
                        st.markdown(f'<div class="source-box">{doc.page_content}</div>', unsafe_allow_html=True)

            # Save assistant response to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": source_docs if show_sources else []
            })

        except Exception as ex:
            err_msg = f"Sorry, an error occurred while generating the answer: {str(ex)}"
            st.error(err_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "sources": []
            })

