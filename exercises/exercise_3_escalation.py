"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()

SYSTEM_PROMPT = """You are a senior software engineer reviewing a GitHub pull request.
Return a structured review: summary, risk_factors, comments (tied to file/line),
confidence in [0, 1], confidence_reasoning, and escalation_questions.

Compute confidence with this rubric (show the arithmetic in confidence_reasoning):
- Start at 0.80.
- SECURITY red flag: subtract 0.15 EACH. Security red flags are ONLY: weak/broken
  password hashing or crypto (MD5, SHA1, no salt), secrets/tokens stored or sent in
  plaintext, SQL/command injection (strings built with f-strings or concatenation),
  missing or bypassable auth, hard-coded credentials or user ids, insecure transport (http).
- MINOR open question: subtract 0.05 EACH, at most 3 of them (max -0.15). Examples:
  schema change without a migration, missing input validation, missing tests,
  unclear default value, naming/UX nits.
- Clamp the result to [0.10, 0.95].

escalation_questions: if your confidence is below 0.60, write 2–4 SPECIFIC questions for
the human reviewer — the things you cannot decide from the diff alone (intent, threat
model, deployment context). Each question must name the file/function it is about and
the concrete concern, e.g. "auth.py `hash_password`: why MD5 instead of bcrypt/argon2 —
is this a placeholder or meant for production?". No generic questions.
If confidence is 0.60 or higher, leave escalation_questions empty.
"""

SYNTH_PROMPT = """You are a senior software engineer finalising a pull-request review.
You already produced an initial review but were unsure, so you asked a human reviewer
some questions. Use their answers to produce a REFINED review:
- Keep issues the answers do not resolve; drop or downgrade ones they clearly resolve;
  add any new issue the answers reveal.
- Comments must be concrete and tied to a file (and line when known).
- Re-score confidence with the same rubric (start 0.80, -0.15 per unresolved security red
  flag, -0.05 per minor open question, max 3). Answers that explain intent raise
  confidence; answers like "don't know" do not.
- In confidence_reasoning, say what changed because of the reviewer's answers.
- Leave escalation_questions empty.
"""


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff'][:40_000]}"},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        # fallback when the LLM didn't generate any questions
        questions = ["What is the intent of this PR?", "Any migration concerns?"]

    console.print("[cyan]→ escalate[/cyan]  [dim]asking reviewer specific questions[/dim]")
    # No side effects before interrupt(): this node re-runs from the top on resume.
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "pr_title": state.get("pr_title", ""),
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    console.print("[cyan]→ synthesize[/cyan]")
    initial = state["analysis"]
    answers = state.get("escalation_answers") or {}
    qa = "\n".join(f"Q: {q}\nA: {ans or '(no answer)'}" for q, ans in answers.items())

    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining the review with reviewer answers...[/dim]"):
        refined = llm.invoke([
            {"role": "system", "content": SYNTH_PROMPT},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n\n"
                f"Diff:\n{state['pr_diff'][:40_000]}\n\n"
                f"--- Initial review (confidence {initial.confidence:.0%}) ---\n"
                f"{initial.model_dump_json(indent=2, exclude={'escalation_questions'})}\n\n"
                f"--- Reviewer Q&A ---\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence {initial.confidence:.0%} → "
                  f"{refined.confidence:.0%}, {len(refined.comments)} comment(s)")
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
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
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        risks = "\n".join(f"• {r}" for r in payload.get("risk_factors", []))
        console.print(Panel.fit(
            escape(f"{payload['summary']}\n\nRisk factors:\n{risks}\n\n{payload['confidence_reasoning']}"),
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"\n[bold yellow]Q:[/bold yellow] {escape(q)}\nA: ").strip()
                for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
