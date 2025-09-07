# backend/app.py
import os
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional: LLM (enabled via USE_OPENAI)
try:
    from langchain_openai import ChatOpenAI
except Exception:  # lib not installed or environment issue
    ChatOpenAI = None  # we'll guard usage

# Optional: GitHub PR support (only used if owner/repo/pr_number provided)
try:
    from backend.github_client import get_pr_diff  # you can add this file later
except Exception:
    get_pr_diff = None

# --- Config ---
load_dotenv()
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Build LLM lazily & safely
llm = None
if USE_OPENAI:
    if not ChatOpenAI:
        raise RuntimeError(
            "langchain-openai not installed but USE_OPENAI=true. Run: pip install langchain-openai"
        )
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY missing but USE_OPENAI=true. Add it to your .env"
        )
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)

# --- FastAPI app ---
app = FastAPI(title="GenAI Code Review")


# --------- Models ----------
class ReviewRequest(BaseModel):
    owner: Optional[str] = None
    repo: Optional[str] = None
    pr_number: Optional[int] = None
    diff: Optional[str] = None
    text: Optional[str] = None
    focus: List[str] = ["security", "performance", "style"]


class Finding(BaseModel):
    category: str
    severity: str
    message: str


class ReviewResult(BaseModel):
    summary: str
    findings: List[Finding]


# --------- Helpers ----------
CHUNK_MAX_CHARS = 8_000


def _split_into_chunks(s: str) -> List[str]:
    if not s:
        return []
    # Prefer diff hunk boundaries @@ ... @@
    parts = re.split(r"(?=^@@.*@@)", s, flags=re.MULTILINE)
    if len(parts) == 1:
        return [s[i : i + CHUNK_MAX_CHARS] for i in range(0, len(s), CHUNK_MAX_CHARS)]
    chunks, current = [], ""
    for p in parts:
        if current and len(current) + len(p) > CHUNK_MAX_CHARS:
            chunks.append(current)
            current = p
        else:
            current += p
    if current:
        chunks.append(current)
    return chunks


SYSTEM_REVIEWER = (
    "You are a senior software reviewer. Be precise, cite concrete lines/snippets from the input, "
    "and focus on actionable suggestions with minimal changes."
)

MAP_PROMPT = """You will analyze part of a code change (diff or source).
Return concise bullet points with: [Issue] → [Why it matters] → [Actionable fix].
Prioritize: {focus_order}.

INPUT CHUNK:
{chunk}
"""

REDUCE_PROMPT = """You will merge multiple partial reviews into a single, non-redundant result.

Focus order: {focus_order}.
1) Write a 5-bullet PR SUMMARY: intent, main changes, risky areas, perf risks, tests impact.
2) Then list FINDINGS grouped by category (Security/Performance/Style).
   For each finding: give short title + 1–2 sentences + severity (High/Med/Low).
3) Keep it crisp and actionable.
PARTIALS:
{partials}
"""


def _focus_order(focus: List[str]) -> str:
    order = ["security", "performance", "style"]
    requested = [f for f in order if f in focus]
    tail = [f for f in order if f not in requested]
    return ", ".join([*requested, *tail])


async def _dummy_review(text: str, focus: List[str]) -> ReviewResult:
    return ReviewResult(
        summary="Dummy review OK. (Set USE_OPENAI=true in .env to enable the real AI reviewer.)",
        findings=[
            Finding(
                category="security",
                severity="High",
                message="Example: potential SQL injection risk.",
            ),
            Finding(
                category="performance",
                severity="Medium",
                message="Example: inefficient loop.",
            ),
            Finding(
                category="style",
                severity="Low",
                message="Example: prefer f-strings over concatenation.",
            ),
        ],
    )


async def _ai_review(text: str, focus: List[str]) -> ReviewResult:
    if not llm:
        raise HTTPException(
            status_code=500,
            detail="LLM not initialized (set USE_OPENAI=true and OPENAI_API_KEY).",
        )

    chunks = _split_into_chunks(text)
    focus_str = _focus_order(focus)

    partials: List[str] = []
    try:
        # MAP phase
        for ch in chunks or ["(empty diff)"]:
            msg = [
                {"role": "system", "content": SYSTEM_REVIEWER},
                {
                    "role": "user",
                    "content": MAP_PROMPT.format(chunk=ch, focus_order=focus_str),
                },
            ]
            resp = await llm.ainvoke(msg)
            partials.append(resp.content.strip())

        # REDUCE phase
        merged_msg = [
            {"role": "system", "content": SYSTEM_REVIEWER},
            {
                "role": "user",
                "content": REDUCE_PROMPT.format(
                    partials="\n\n---\n\n".join(partials), focus_order=focus_str
                ),
            },
        ]
        final_resp = await llm.ainvoke(merged_msg)
        merged_text = final_resp.content.strip()
    except Exception as e:
        # Surface real OpenAI/LLM errors
        raise HTTPException(
            status_code=502, detail=f"LLM call failed: {type(e).__name__}: {e}"
        )

    # Minimal parse: try to pull a bullet summary; fallback to first 800 chars
    summary_match = re.search(
        r"(?:SUMMARY|Summary).*?\n(?P<body>(?:- .*\n?){3,8})",
        merged_text,
        re.IGNORECASE | re.DOTALL,
    )
    summary = (
        summary_match.group("body").strip() if summary_match else merged_text[:800]
    )

    findings: List[Finding] = []
    for cat in ["Security", "Performance", "Style"]:
        m = re.search(
            rf"{cat}.*?(?:\n- .*(?:\n- .*)*)",
            merged_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            continue
        bullets = re.findall(r"-\s*(.*)", m.group(0))
        for b in bullets[:6]:
            sev = (
                "High"
                if re.search(r"\b(high|critical|severe)\b", b, re.I)
                else (
                    "Medium" if re.search(r"\b(medium|moderate)\b", b, re.I) else "Low"
                )
            )
            findings.append(
                Finding(category=cat.lower(), severity=sev, message=b.strip())
            )

    return ReviewResult(summary=summary, findings=findings)


# --------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/review", response_model=ReviewResult)
async def review(req: ReviewRequest):
    """
    Provide either:
      - diff or text
      - OR owner/repo/pr_number (requires backend/github_client.py + GITHUB_TOKEN)
    """
    # Choose input source
    src = None
    if req.owner and req.repo and req.pr_number:
        if not get_pr_diff:
            raise HTTPException(
                status_code=500,
                detail="GitHub integration not available (missing backend/github_client.py)",
            )
        try:
            src = get_pr_diff(req.owner, req.repo, req.pr_number)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch PR diff: {type(e).__name__}: {e}",
            )
    else:
        src = req.diff or req.text

    if not src:
        raise HTTPException(
            status_code=400, detail="Provide a diff/text OR (owner, repo, pr_number)."
        )

    # Dispatch to dummy or AI reviewer
    try:
        if USE_OPENAI:
            return await _ai_review(src, req.focus)
        else:
            return await _dummy_review(src, req.focus)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Review failed: {type(e).__name__}: {e}"
        )
