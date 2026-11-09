"""LangGraph agent: plan -> retrieve -> answer -> verify.

Genuinely agentic rather than a bare RAG chain: the plan step decides what to search for
and whether a structured metadata lookup applies, and the verify step is a hard gate —
an answer without a citation back to a retrieved chunk is replaced with a refusal instead
of being returned to the user.
"""

from __future__ import annotations

import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from policypilot.agent.llm import LLMClient
from policypilot.agent.tools import filing_metadata, format_context, retrieve
from policypilot.config import DEFAULT_TICKERS
from policypilot.retrieval.base import SearchResult, VectorStore

CITATION_RE = re.compile(r"\[(\d+)\]")

# Variants models produce instead of [n]: 【2】 ［2］ 〔2〕 【2†source】 [1, 2]
_ANY_CITATION_RE = re.compile(
    r"[\[【［〔]\s*(\d+(?:\s*[,，]\s*\d+)*)\s*(?:†[^\]】］〕]*)?[\]】］〕]"
)

MIN_SIMILARITY = 0.35

USE_GROUNDEDNESS_JUDGE = False

JUDGE_SYSTEM_PROMPT = (
    "You check answers for groundedness. Reply with ONLY one word: YES if every factual "
    "claim in the ANSWER is supported by the CONTEXT, otherwise NO."
)


PLAN_SYSTEM_PROMPT = (
    "You turn a user's question about SEC 10-K filings into a short search query "
    "for a semantic search index over filing text, and detect which company (if any) "
    "the question is about. Known companies (by ticker): " + ", ".join(DEFAULT_TICKERS) + ". "
    "The question may name a company by its ticker symbol OR by its common name (e.g. "
    "'Apple' means AAPL) — resolve either form to the ticker symbol.\n\n"
    "Reply with ONLY a JSON object of the form "
    '{"search_query": "<short search query text>", "ticker": "<TICKER or null>"}, '
    "nothing else."
)

ANSWER_SYSTEM_PROMPT = (
    "You are PolicyPilot, a compliance research assistant that answers questions about "
    "SEC 10-K filings using ONLY the numbered context provided below. This is a strict "
    "citation requirement: every sentence containing a factual claim MUST end with a "
    "bracketed citation number like [1] or [2] matching the context item it came from — "
    "no other citation style (no footnotes, no parentheses, no 'Source:' lines). "
    "Example: 'The filing discloses reliance on a limited number of suppliers [1].' "
    "If the context does not contain enough information to answer, say so plainly instead "
    "of guessing — never answer from general knowledge.\n\nContext:\n{context}"
)

REFUSAL_MESSAGE = (
    "I don't have enough grounded information in the ingested filings to answer that "
    "confidently. Try rephrasing, or ask about one of the ingested companies."
)


class AgentState(TypedDict, total=False):
    question: str
    search_query: str
    ticker_hint: str | None
    results: list[SearchResult]
    metadata: dict | None
    draft_answer: str
    final_answer: str
    grounded: bool
    verify_reason: str


def normalize_citations(text: str) -> str:
    """Rewrite citation variants into plain [n] so verification and the UI agree."""

    def _repl(m: re.Match) -> str:
        return "".join(f"[{n}]" for n in re.split(r"\s*[,，]\s*", m.group(1)))

    return _ANY_CITATION_RE.sub(_repl, text)


def _cosine(score: float) -> float:
    """LocalChromaVectorStore's score is 1 - squared L2 distance = 2*cos - 1 for normalized
    vectors, so convert back to cosine. If the collection is switched to cosine distance
    (hnsw:space='cosine'), the score already IS cosine — then return score unchanged."""
    return (score + 1) / 2


def build_graph(llm: LLMClient, store: VectorStore):
    def plan_node(state: AgentState) -> AgentState:
        logger.info("Planning search for question: {!r}", state["question"])
        reply = llm.complete(
            system=PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["question"]}],
        ).strip()
        try:
            parsed = json.loads(reply)
            query = str(parsed.get("search_query") or "").strip()
            ticker = parsed.get("ticker")
            ticker = str(ticker).strip().upper() if ticker else None
        except (json.JSONDecodeError, AttributeError):
            query, ticker = reply, None
        ticker_hint = ticker if ticker in DEFAULT_TICKERS else None
        return {"search_query": query or state["question"], "ticker_hint": ticker_hint}

    def retrieve_node(state: AgentState) -> AgentState:
        logger.info(
            "Retrieving chunks for query {!r} (ticker hint: {})",
            state["search_query"],
            state.get("ticker_hint"),
        )
        ticker_hint = state.get("ticker_hint")
        results = retrieve(store, state["search_query"], k=3, ticker=ticker_hint)
        metadata = filing_metadata(ticker_hint) if ticker_hint else None
        logger.info(
            "Retrieved {} chunks for query {!r} (ticker hint: {})",
            len(results),
            state["search_query"],
            ticker_hint,
        )
        logger.info("metadata for ticker hint {}: {}", ticker_hint, metadata)
        return {"results": results, "metadata": metadata}

    def answer_node(state: AgentState) -> AgentState:
        logger.info("Generating answer for question: {!r}", state["question"])
        context = format_context(state["results"])
        logger.info("Formatted context for answer generation: {}", context)
        if not context:
            return {"draft_answer": REFUSAL_MESSAGE}
        system = ANSWER_SYSTEM_PROMPT.format(context=context)
        draft = llm.complete(
            system=system, messages=[{"role": "user", "content": state["question"]}]
        )
        draft = normalize_citations(draft)
        logger.info("Draft answer generated: {}", draft)
        return {"draft_answer": draft}

    def verify_node(state: AgentState) -> AgentState:
        logger.info("Verifying answer for question: {!r}", state["question"])
        draft = state["draft_answer"]
        results = state.get("results") or []

        def refuse(reason: str) -> AgentState:
            logger.warning("verify REFUSED: {} | draft: {}", reason, draft[:300])
            return {"final_answer": REFUSAL_MESSAGE, "grounded": False, "verify_reason": reason}

        # 1. Anything retrieved at all?
        if not results:
            return refuse("no chunks retrieved")

        # 2. Is the best chunk actually relevant to the question?
        best = max(_cosine(r.score) for r in results)
        logger.info("verify: best_similarity={:.3f} (threshold {})", best, MIN_SIMILARITY)
        if best < MIN_SIMILARITY:
            return refuse(f"best chunk similarity {best:.3f} below {MIN_SIMILARITY}")

        # 3 + 4. At least one citation, and every [n] must point to a real chunk.
        cited = {int(n) for n in CITATION_RE.findall(draft)}
        if not cited:
            return refuse("answer has no citations")
        invalid = sorted(n for n in cited if not 1 <= n <= len(results))
        if invalid:
            return refuse(f"citations {invalid} don't match any of the {len(results)} chunks")

        # 5. Optional: does the context actually support the answer?
        if USE_GROUNDEDNESS_JUDGE:
            verdict = (
                llm.complete(
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": f"CONTEXT:\n{format_context(results)}\n\nANSWER:\n{draft}",
                        }
                    ],
                )
                .strip()
                .upper()
            )
            logger.info("verify: judge verdict={}", verdict)
            if not verdict.startswith("YES"):
                return refuse(f"groundedness judge said {verdict!r}")

        logger.info("verify PASSED: citations {} of {} chunks", sorted(cited), len(results))
        return {"final_answer": draft, "grounded": True, "verify_reason": "passed"}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def ask(llm: LLMClient, store: VectorStore, question: str) -> AgentState:
    logger.info("Asking question ask method: {!r}", question)
    app = build_graph(llm, store)
    return app.invoke({"question": question})
