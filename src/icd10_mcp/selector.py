"""
ICD-10-CM code selector — picks best codes from retriever candidates.
Adapted from your original selector.py: only the import path changed.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from icd10_mcp.openrouter_client import chat_completion

logger = logging.getLogger(__name__)


SELECTOR_SYSTEM = """You are an ICD-10-CM coding assistant.

You will be given:
1) A clinical note
2) Extracted problems
3) Candidate ICD-10-CM codes (titles + metadata) from retrieval

Task:
- For EACH problem, select the best code(s) using ONLY the provided candidates.
- Do NOT invent codes not in candidates.
- Prefer the MOST specific code supported by the note and candidate title.
- Keep output stable, minimal, and evidence-based.

SPECIFICITY HIERARCHY -- apply in this strict order:
  1. Most specific combination code covering co-existing conditions (if in candidates)
  2. Most specific single code fully supported by the note
  3. "Other specified" code (titles containing "other" or ending in patterns like .09, .89, .x9)
     when the condition is documented but the exact sub-type is not named in the note
  4. "Unspecified" code only as last resort when the note truly lacks the required detail
- NEVER select both a parent code and its child code for the same condition.
- NEVER select an "unspecified" code when a more specific candidate is supported by the note.

PARENT CODE PROHIBITION:
- A code is a PARENT of another when its value is a strict alphanumeric prefix of the other.
- When candidates include BOTH a parent code AND child codes, ALWAYS select the child.

NAMED DISEASE vs RESIDUAL CATEGORY:
- When candidates contain BOTH a specific named disease code AND a generic residual code,
  prefer the specific named disease code if the note matches, even if the terminology differs.

DIAGNOSIS CODE vs FINDING/SYMPTOM CODE:
- When candidates contain BOTH an R-chapter code AND a disease-chapter code for the same
  condition: ALWAYS select the disease-chapter code.

MISSING DOCUMENTATION:
- When required detail is absent, select the "unspecified" candidate and note what is missing.

COMBINATION AND ETIOLOGY-MANIFESTATION CODING:
- Check candidates for a combination code covering both primary + manifestation. Prefer it.
- When no combination code exists, code both individually.

DUPLICATE PREVENTION:
- The same code must never appear more than once across all problems.

Return STRICT JSON only:
{
  "results": [
    {
      "problem": "string",
      "selected_codes": [
        {"code": "string", "title": "string", "rationale": "short evidence-based reason"}
      ],
      "notes": "optional short note"
    }
  ]
}
"""


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _safe_json_loads(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Salvage any complete result objects from a truncated response
    salvaged: List[Dict[str, Any]] = []
    for m in re.finditer(r'\{[^{}]*"problem"[^{}]*\}', cleaned, re.DOTALL):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and obj.get("problem"):
                salvaged.append(obj)
        except json.JSONDecodeError:
            pass
    if salvaged:
        return {"results": salvaged}

    return {"results": []}


def _is_unspecified_title(title: str) -> bool:
    t = (title or "").lower()
    return "unspecified" in t or "unspec" in t


def _drop_ancestors_and_unspecified(selected: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not selected:
        return selected

    codes = [s.get("code", "").strip() for s in selected if s.get("code")]
    titles = {s.get("code", "").strip(): (s.get("title", "") or "") for s in selected}

    def category(c: str) -> str:
        return c.split(".")[0].strip()

    drop = set()

    for c in codes:
        for o in codes:
            if c == o:
                continue
            if o.startswith(c) and len(o) > len(c):
                drop.add(c)

    for c in codes:
        if _is_unspecified_title(titles.get(c, "")):
            cat = category(c)
            if any(
                (o != c) and (category(o) == cat) and (not _is_unspecified_title(titles.get(o, "")))
                for o in codes
            ):
                drop.add(c)

    return [s for s in selected if (s.get("code", "").strip() not in drop)]


def _global_dedupe(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []

    for r in results:
        codes = r.get("selected_codes") or []
        if not isinstance(codes, list):
            out.append(r)
            continue

        kept = []
        dropped = []
        for c in codes:
            code = (c.get("code") or "").strip()
            if not code:
                continue
            if code in seen:
                dropped.append(code)
            else:
                seen.add(code)
                kept.append(c)

        if dropped:
            note = (r.get("notes") or "").strip()
            extra = f"duplicate code removed across problems: {', '.join(dropped)}"
            r["notes"] = (note + ("; " if note else "") + extra).strip()

        r["selected_codes"] = kept
        out.append(r)

    return out


def select_codes(
    *,
    note: str,
    planned_problems: List[Dict[str, Any]],
    merged_candidates: List[Dict[str, Any]],
    openrouter_api_key: str,
    model: str,
    max_codes_per_problem: int = 2,
    candidates_by_problem: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:

    def to_candidate_rows(cands: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for c in cands[:limit]:
            md = c.get("metadata") or {}
            code = c.get("code") or c.get("id") or ""
            title = c.get("title", "") or (md.get("title", "") if isinstance(md, dict) else "")
            out_rows.append(
                {
                    "code": code,
                    "title": title,
                    "score": round(float(c.get("score", 0.0)), 6),
                    "parent_code": md.get("parent_code", "") if isinstance(md, dict) else "",
                    "level": md.get("level", None) if isinstance(md, dict) else None,
                    "problems": c.get("problems", []),
                }
            )
        return [r for r in out_rows if r.get("code")]

    if candidates_by_problem:
        payload_candidates: Dict[str, List[Dict[str, Any]]] = {}
        for p in planned_problems:
            prob = (p.get("problem") or "").strip()
            if not prob:
                continue
            cands = candidates_by_problem.get(prob, [])
            payload_candidates[prob] = to_candidate_rows(cands, limit=min(len(cands), 20))

        conf_order = {"high": 0, "medium": 1, "low": 2}
        sorted_problems = sorted(
            planned_problems,
            key=lambda p: conf_order.get((p.get("confidence") or "low").lower(), 2),
        )
        top_problems = sorted_problems[:12]
        top_prob_names = {(p.get("problem") or "").strip() for p in top_problems}
        payload_candidates = {k: v for k, v in payload_candidates.items() if k in top_prob_names}

        user_payload = {
            "note": note,
            "max_codes_per_problem": max_codes_per_problem,
            "planned_problems": top_problems,
            "candidates_by_problem": payload_candidates,
        }
    else:
        top_candidates = merged_candidates[:80]
        user_payload = {
            "note": note,
            "max_codes_per_problem": max_codes_per_problem,
            "planned_problems": planned_problems,
            "candidates": to_candidate_rows(top_candidates, limit=80),
        }

    payload_json = json.dumps(user_payload, ensure_ascii=False)
    logger.info(
        "Selector: sending %d-char payload (%d problems) to %s",
        len(payload_json),
        len(user_payload.get("planned_problems", [])),
        model,
    )

    text = chat_completion(
        model=model,
        api_key=openrouter_api_key,
        messages=[
            {"role": "system", "content": SELECTOR_SYSTEM},
            {"role": "user", "content": payload_json},
        ],
        temperature=0.0,
        max_tokens=4000,
    )
    logger.info("Selector: raw response (%d chars): %.500s", len(text), text)

    data = _safe_json_loads(text)
    results = data.get("results", [])
    cleaned_results: List[Dict[str, Any]] = []

    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue

        problem = (item.get("problem") or "").strip()
        selected_codes = item.get("selected_codes") or []
        notes = (item.get("notes") or "").strip()

        if isinstance(selected_codes, str) and selected_codes.lower().strip() == "needs clarification":
            cleaned_results.append({"problem": problem, "selected_codes": [], "notes": notes or "Needs clarification"})
            continue

        if not isinstance(selected_codes, list):
            selected_codes = []

        norm = []
        for s in selected_codes:
            if not isinstance(s, dict):
                continue
            code = (s.get("code") or "").strip()
            title = (s.get("title") or "").strip()
            rationale = (s.get("rationale") or "").strip()
            if code:
                norm.append({"code": code, "title": title, "rationale": rationale})

        norm = _drop_ancestors_and_unspecified(norm)
        norm = norm[:max_codes_per_problem]

        cleaned_results.append({"problem": problem, "selected_codes": norm, "notes": notes})

    cleaned_results = _global_dedupe(cleaned_results)
    return {"results": cleaned_results}