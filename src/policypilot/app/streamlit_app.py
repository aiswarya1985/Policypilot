"""PolicyPilot chat UI. Local: `streamlit run src/policypilot/app/streamlit_app.py`.
In production this becomes the Databricks App (see resources/apps.yml) — same code,
deployed behind an app service principal with on-behalf-of-user auth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policypilot.agent.graph import ask
from policypilot.agent.llm import get_llm_client
from policypilot.config import DEFAULT_TICKERS, get_settings
from policypilot.ingestion.pipeline import get_vector_store
from policypilot.utilities.logger import setup_logger

st.set_page_config(page_title="PolicyPilot", page_icon="📄")
st.title("📄 PolicyPilot")
st.caption(
    "Governed Q&A over SEC 10-K filings — every answer cites the filing it came from, "
    "or refuses if it can't."
)

@st.cache_resource
def _init_logging() -> bool:
    setup_logger("DEBUG")
    logger.info("PolicyPilot app starting")
    return True


_init_logging()

@st.cache_resource
def _load_backend():
    settings = get_settings()
    logger.info("Loading backend (PP_ENV={})", settings.env)
    if settings.is_local and not settings.groq_api_key:
        logger.error("GROQ_API_KEY missing")
        return None, None, settings
    llm, store = get_llm_client(), get_vector_store()
    logger.info("Backend ready: {} + {}", type(llm).__name__, type(store).__name__)
    return llm, store, settings

llm, store, settings = _load_backend()

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"Tickers ingested: {', '.join(DEFAULT_TICKERS)}")
    st.caption("Run `uv run python -m policypilot.ingestion.pipeline` to (re)ingest.")
    st.subheader("Backend")
    st.write(f"PP_ENV = `{settings.env}`")

if llm is None:
    st.error(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, then restart."
    )
    st.stop()

def _render_sources(results) -> None:
    """Map each [n] citation in the answer back to the filing it actually came from."""
    if not results:
        return
    with st.expander(f"Sources ({len(results)})"):
        for i, r in enumerate(results, start=1):
            m = r.metadata or {}
            st.markdown(
                f"**[{i}] {m.get('company', 'Unknown')}** — 10-K filed "
                f"{m.get('filing_date', 'unknown date')} "
                f"(accession {m.get('accession_number', 'unknown')})"
            )
            st.caption(r.text[:300] + ("…" if len(r.text) > 300 else ""))


if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        _render_sources(turn.get("sources"))

if question := st.chat_input("Ask a question about the ingested filings..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            logger.info("Question: {!r}", question)
            result = ask(llm, store, question)
        answer = result["final_answer"]
        grounded = result.get("grounded")
        results = result.get("results") if grounded else None
        st.markdown(answer)
        if not grounded:
            st.caption("⚠️ No grounded citation found — refused rather than guessing.")
        _render_sources(results)
    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": results}
    )
