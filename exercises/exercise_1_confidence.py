"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()

MAX_DIFF_CHARS = 40_000  # keep the prompt bounded on large PRs

SYSTEM_PROMPT = """You are a senior software engineer reviewing a GitHub pull request.
Read the diff and return a structured review:
- summary: one paragraph on what the PR does.
- risk_factors: concrete risks (security, data migration, missing tests, breaking changes...).
- comments: specific review comments tied to a file (and line when you can tell).
- confidence: a calibrated number in [0, 1] — how confident you are that your review is
  complete and correct AND that the PR can be merged safely as-is.
- confidence_reasoning: why you picked that number.

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

Guidance: a small feature with only minor open questions lands around 0.65–0.70;
any PR with two or more security red flags must land well below 0.58.
"""


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human",
         f"PR title: {state.get('pr_title', '')}\n"
         f"Files changed: {', '.join(state.get('pr_files', []))}\n\n"
         f"Unified diff:\n```diff\n{state.get('pr_diff', '')[:MAX_DIFF_CHARS]}\n```"),
    ]
    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke(messages)
    console.print(f"  [green]✓[/green] confidence {analysis.confidence:.0%} — "
                  f"{len(analysis.comments)} comments, {len(analysis.risk_factors)} risk factors")
    return {"analysis": analysis}


def decide(confidence: float) -> str:
    """Map a confidence score to a branch name."""
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "auto_approve"
    if confidence < ESCALATE_THRESHOLD:
        return "escalate"
    return "human_approval"


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    decision = decide(state["analysis"].confidence)
    console.print(f"  [green]✓[/green] decision = {decision}")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    for terminal in ("auto_approve", "human_approval", "escalate"):
        g.add_edge(terminal, END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
