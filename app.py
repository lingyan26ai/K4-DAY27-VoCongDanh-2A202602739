"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from common.db import db_conn, db_path
from common.schemas import AuditEntry, ReviewState, risk_level_for
from exercises.exercise_4_audit import (
    AGENT_ID,
    _already_logged,
    _reviewer_id,
    audit,
    node_analyze,
    node_auto_approve,
    node_commit,
    node_escalate,
    node_fetch_pr,
    node_human_approval,
    node_route,
    node_synthesize,
)


load_dotenv()


# ─── Graph: exercise 4 nodes + a final confirm after escalation ────────────
async def node_confirm_refined(state):
    """After synthesize: show the refined review and let the reviewer confirm.

    Returning human_choice="reject" and clearing escalation_answers makes the
    exercise-4 `node_commit` skip posting.
    """
    t0 = time.monotonic()
    a = state["analysis"]
    if not await _already_logged(state, "confirm_refined", "pending"):
        await audit(state, AuditEntry(
            agent_id=AGENT_ID, action="confirm_refined",
            confidence=a.confidence, risk_level=risk_level_for(a.confidence),
            decision="pending", reason="Refined review shown for final confirm",
            execution_time_ms=int((time.monotonic() - t0) * 1000),
        ))
    resp = interrupt({
        "kind": "refined_review",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "answers": state.get("escalation_answers") or {},
    })
    choice = resp.get("choice", "reject")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID, action="confirm_refined",
        confidence=a.confidence, risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(), decision=choice,
        reason=resp.get("feedback") or f"Reviewer chose {choice} on refined review",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    if choice == "approve":
        return {"human_feedback": resp.get("feedback") or None}
    return {"human_choice": "reject", "escalation_answers": None,
            "human_feedback": resp.get("feedback") or None}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("escalate", node_escalate), ("synthesize", node_synthesize),
        ("confirm_refined", node_confirm_refined), ("commit", node_commit),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "confirm_refined")
    g.add_edge("confirm_refined", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


async def run_graph(pr_url: str, thread_id: str, resume_value=None) -> dict:
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str) -> tuple[dict, dict | None]:
    """Read a saved thread from the checkpointer: (state values, pending interrupt payload)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    pending = [i.value for t in snap.tasks for i in (t.interrupts or ())]
    return dict(snap.values or {}), (pending[0] if pending else None)


def recent_threads(limit: int = 15) -> list[dict]:
    """Recent sessions from audit_events (same idea as audit.replay.list_threads)."""
    try:
        conn = sqlite3.connect(db_path())
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT thread_id, pr_url,
                   MAX(timestamp) AS last_event,
                   MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END) AS worst,
                   (SELECT action || ':' || decision FROM audit_events e2
                     WHERE e2.thread_id = e.thread_id ORDER BY id DESC LIMIT 1) AS last_step
              FROM audit_events e
             GROUP BY thread_id, pr_url
             ORDER BY MAX(id) DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:  # db / table not created yet
        return []


# ─── Session state ─────────────────────────────────────────────────────────
for key, default in [("thread_id", None), ("pr_url", ""), ("interrupt_payload", None),
                     ("final", None), ("error", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def open_thread(thread_id: str) -> None:
    """Load a saved session (sidebar click or ?thread=... after a page reload)."""
    values, pending = asyncio.run(load_thread(thread_id))
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = values.get("pr_url", "")
    st.session_state.interrupt_payload = pending
    st.session_state.final = None if pending else (values or None)
    st.session_state.error = None
    st.query_params["thread"] = thread_id


def apply_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


# Browser reload wipes st.session_state; the thread id in the URL brings it back.
if st.session_state.thread_id is None and st.query_params.get("thread"):
    try:
        open_thread(st.query_params["thread"])
    except Exception as e:  # noqa: BLE001
        st.session_state.error = f"Could not reload thread: {e}"


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")

RISK_BADGE = {3: "🔴 high", 2: "🟡 med", 1: "🟢 low"}


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    threads = recent_threads()
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr_no = t["pr_url"].rstrip("/").split("/")[-1]
        current = t["thread_id"] == st.session_state.thread_id
        label = (f"{'▶ ' if current else ''}PR #{pr_no} · {RISK_BADGE.get(t['worst'], '?')}\n\n"
                 f"{t['last_step']} · {t['last_event'][:19].replace('T', ' ')}")
        if st.button(label, key=f"thr_{t['thread_id']}", use_container_width=True):
            try:
                open_thread(t["thread_id"])
            except Exception as e:  # noqa: BLE001
                st.session_state.error = f"Could not open thread: {e}"
            st.rerun()
    if st.session_state.thread_id:
        st.divider()
        st.caption(f"Current thread\n\n`{st.session_state.thread_id}`")
        if st.button("New review", use_container_width=True):
            for k in ("thread_id", "interrupt_payload", "final", "error"):
                st.session_state[k] = None
            st.query_params.clear()
            st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def _comments(comments: list[dict]) -> None:
    icon = {"blocker": "⛔", "issue": "🔴", "suggestion": "🟡", "nit": "⚪"}
    for c in comments:
        st.markdown(f"- {icon.get(c['severity'], '•')} **[{c['severity']}]** "
                    f"`{c['file']}:{c.get('line') or '?'}` — {c['body']}")


def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"🟡 Approval requested — confidence {conf:.0%}")
    st.progress(conf)
    st.markdown(payload["summary"])
    with st.expander("Why this confidence?"):
        st.write(payload["confidence_reasoning"])

    st.markdown("**Proposed comments**")
    _comments(payload.get("comments", []))

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    tid = st.session_state.thread_id
    feedback = st.text_input("Feedback (optional — required for Edit)", key=f"approval_feedback_{tid}")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True, key=f"approve_{tid}"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True, key=f"reject_{tid}"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", use_container_width=True, key=f"edit_{tid}"):
        if not feedback.strip():
            st.warning("Write your edit in the Feedback box first — it is appended to the posted review.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🔴 Strong escalation — confidence {conf:.0%}")
    st.progress(conf)
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])
    with st.expander("Why this confidence?"):
        st.write(payload["confidence_reasoning"])

    st.markdown("**The agent needs your input on these questions:**")
    tid = st.session_state.thread_id
    with st.form(f"escalation_{tid}"):
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_area(f"**Q{i + 1}.** {q}", key=f"q{i}_{tid}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            return {q: (a.strip() or "(no answer)") for q, a in answers.items()}
    return None


def render_refined_card(payload: dict) -> dict | None:
    """After escalation: show the refined review for a final confirm."""
    conf = payload["confidence"]
    st.subheader(f"Refined review — confidence {conf:.0%}")
    st.progress(conf)
    st.markdown(payload["summary"])
    with st.expander("What changed after your answers"):
        st.write(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.warning("**Remaining risks**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown("**Comments to post**")
    _comments(payload.get("comments", []))
    with st.expander("Your answers"):
        for q, a in payload.get("answers", {}).items():
            st.markdown(f"**{q}**\n\n> {a}")

    tid = st.session_state.thread_id
    feedback = st.text_input("Extra note (optional)", key=f"refined_feedback_{tid}")
    col1, col2 = st.columns(2)
    if col1.button("✅ Post refined review", type="primary", use_container_width=True, key=f"post_{tid}"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Discard", use_container_width=True, key=f"discard_{tid}"):
        return {"choice": "reject", "feedback": feedback}
    return None


RENDERERS = {
    "approval_request": render_approval_card,
    "escalation": render_escalation_card,
    "refined_review": render_refined_card,
}


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None
    st.query_params["thread"] = st.session_state.thread_id

    with st.spinner("Fetching PR + asking the LLM..."):
        try:
            apply_result(asyncio.run(run_graph(pr_url, st.session_state.thread_id)))
        except Exception as e:  # noqa: BLE001
            st.session_state.error = f"{type(e).__name__}: {e}"
    st.rerun()

if st.session_state.error:
    st.error(st.session_state.error)

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    answer = RENDERERS[payload["kind"]](payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            try:
                apply_result(asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                )))
                st.session_state.error = None
            except Exception as e:  # noqa: BLE001
                st.session_state.error = f"{type(e).__name__}: {e}"
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action") or "?"
    a = final.get("analysis")
    pr = st.session_state.pr_url

    if action.startswith("auto"):
        st.success(f"🟢 **Auto-approved** — confidence {a.confidence:.0%}. "
                   "The agent posted the review itself; nothing for you to do.")
    elif action.startswith("committed"):
        st.success(f"✅ **{action}** — review posted to the PR.")
    elif action == "rejected":
        st.warning("Rejected — no comment posted.")
    elif "failed" in action:
        st.error(f"{action} — check GITHUB_TOKEN scope (needs `public_repo`).")
    else:
        st.info(f"final_action = {action}")

    if a is not None:
        st.markdown(f"**Summary** (confidence {a.confidence:.0%})\n\n{a.summary}")
        _comments([c.model_dump() for c in a.comments])
    if pr and action != "rejected":
        st.link_button("View comment on GitHub", pr)
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
