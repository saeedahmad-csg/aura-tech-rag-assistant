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

# Initialize Theme State (default: light)
if "theme" not in st.session_state:
    st.session_state.theme = "light"

is_dark = st.session_state.theme == "dark"

# Dynamic Theme CSS Variables & Rules
if is_dark:
    theme_css = """
    /* Dark Mode Overrides */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background-color: #0F172A !important;
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4) !important;
    }
    .hero-title {
        color: #F8FAFC !important;
    }
    .hero-subtitle {
        color: #94A3B8 !important;
    }
    .sidebar-brand-card {
        background: #0F172A !important;
        border: 1px solid #334155 !important;
    }
    .brand-title {
        color: #F8FAFC !important;
    }
    div[data-testid="column"] button {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #E2E8F0 !important;
    }
    div[data-testid="column"] button:hover {
        background: #334155 !important;
        color: #60A5FA !important;
        border-color: #60A5FA !important;
    }
    .source-box {
        background-color: #1E293B !important;
        border-left: 3px solid #60A5FA !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stChatMessage"] * {
        color: #F8FAFC !important;
    }
    div[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="base-input"], div[data-baseweb="input"], textarea {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"] button, [data-testid="collapsedControl"] button {
        color: #F8FAFC !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
    }
    """
else:
    theme_css = """
    /* Light Mode Overrides */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .hero-title {
        color: #0F172A;
    }
    .hero-subtitle {
        color: #475569;
    }
    .sidebar-brand-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
    }
    .brand-title {
        color: #0F172A;
    }
    """

# Custom CSS for clean, modern enterprise UI styling
st.markdown(f"""
<style>
    /* Remove any hiding of Streamlit header so the sidebar expand button (>>) is always visible */
    [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }}

    /* Hide the Deploy button in the top right */
    .stDeployButton, [data-testid="stDeployButton"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* In the MainMenu dropdown: keep only the Theme Selector (first child) and hide everything else */
    ul[data-testid="main-menu-list"] > li:not(:first-child),
    ul[data-testid="main-menu-list"] > div,
    ul[data-testid="main-menu-list"] ~ div {{
        display: none !important;
    }}

    /* Base Container Max Width */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 950px;
    }}

    /* Hero Banner Card */
    .hero-banner {{
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }}
    .hero-title {{
        font-size: 1.85rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }}
    .hero-subtitle {{
        font-size: 0.95rem;
        line-height: 1.5;
        margin: 0;
    }}

    /* Sidebar Branding */
    .sidebar-brand-card {{
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: center;
    }}
    .brand-title {{
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 8px;
        margin-bottom: 2px;
    }}

    /* Quick question chips */
    .quick-section-title {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
        opacity: 0.75;
    }}
    div[data-testid="column"] button {{
        border-radius: 10px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease-in-out !important;
        min-height: 44px !important;
    }}
    div[data-testid="column"] button:hover {{
        transform: translateY(-2px);
    }}

    /* Source citations styling */
    .source-box {{
        padding: 10px 14px;
        margin-top: 6px;
        border-radius: 6px;
        font-size: 0.88rem;
        line-height: 1.5;
    }}

    /* ======================================================== */
    /* Mobile & Android Responsive Breakpoints (<= 768px)       */
    /* ======================================================== */
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
            padding-bottom: 5rem !important;
            max-width: 100% !important;
        }}
        .hero-banner {{
            padding: 16px 18px !important;
            border-radius: 12px !important;
            margin-bottom: 16px !important;
        }}
        .hero-title {{
            font-size: 1.35rem !important;
            gap: 8px !important;
        }}
        .hero-subtitle {{
            font-size: 0.85rem !important;
            line-height: 1.4 !important;
        }}
        /* Make 4 quick-question columns wrap smoothly into 2x2 grid on mobile */
        div[data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 8px !important;
        }}
        div[data-testid="column"] {{
            flex: 1 1 45% !important;
            min-width: 140px !important;
        }}
        div[data-testid="column"] button {{
            font-size: 0.8rem !important;
            padding: 8px 10px !important;
            width: 100% !important;
        }}
        .stChatMessage {{
            padding: 0.4rem 0.6rem !important;
            font-size: 0.92rem !important;
        }}
        /* Mobile bottom safe-area & floating keyboard support */
        [data-testid="stBottom"] {{
            padding-bottom: env(safe-area-inset-bottom, 8px) !important;
        }}
        [data-testid="collapsedControl"] {{
            top: 0.5rem !important;
            left: 0.5rem !important;
        }}
    }}

    {theme_css}
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

    # Generate Assistant Response (Real-time Streaming)
    with st.chat_message("assistant"):
        try:
            stream_gen, source_docs = rag.stream_rag(
                query=effective_prompt,
                vectorstore=vectorstore,
                api_key=active_key,
                model_name=selected_model,
                k=3,
                chat_history=st.session_state.messages[:-1]
            )
            
            # Stream response in real-time
            answer_text = st.write_stream(stream_gen)

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

