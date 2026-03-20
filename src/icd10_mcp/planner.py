"""
Clinical query planner — extracts coding problems from a clinical note.
Adapted from your original planner.py: only the import path changed.
"""
import json
import logging
import re
from typing import Any, Dict, List

from icd10_mcp.openrouter_client import chat_completion

logger = logging.getLogger(__name__)


PLANNER_SYSTEM = """You are a clinical query planner for ICD-10-CM retrieval.

You will receive a structured clinical lab/investigation report and output STRICT JSON
with a de-duplicated list of coding-relevant clinical problems.

The report is structured as:
  INVESTIGATIONS       -- list of report names, dates, specimen types
  ABNORMAL RESULTS     -- flagged findings (HIGH / LOW / CRITICAL / ABNORMAL). Use these for coding.
  NORMAL/WITHIN-RANGE  -- context only; do not generate problems from these unless unique.

OUTPUT FORMAT (STRICT):
{
  "problems": [
    {
      "problem": "string",
      "confidence": "high|medium|low",
      "queries": ["string", "..."]
    }
  ]
}

CORE RULES:
- Output JSON only. No markdown. No commentary.
- Do NOT invent diagnoses not supported by the report.
- Derive problems primarily from the ABNORMAL RESULTS section.
- Prefer the clinical diagnosis name over a description of the measurement.
- Keep the MOST specific concept only -- no duplicates at different specificity levels.
- Stable ordering:
  1) Primary diagnoses (directly named or clearly supported by abnormal results)
  2) Complications and manifestations of primary diagnoses in other organs
  3) Co-morbid conditions supported by abnormal results
  4) Isolated abnormal findings only when no diagnosis name can be concluded

DIAGNOSIS ELEVATION:
- Each abnormal result is a clue to an underlying condition -- not simply a reading.
- For each flagged result ask: what named clinical condition does this value support?
  Generate a problem using THAT CONDITION NAME, not the measurement name.
  Examples of the principle (applies to ANY condition, not just these):
    HIGH creatinine + LOW eGFR        -> problem: chronic kidney disease (include stage -- see CKD STAGING)
    HIGH aldosterone or LOW renin     -> problem: primary hyperaldosteronism
    HIGH cholesterol + HIGH LDL       -> problem: hyperlipidemia or mixed dyslipidemia
    HIGH homocysteine                 -> problem: hyperhomocysteinemia
    HIGH CRP or elevated ESR          -> problem: elevated inflammatory marker
    HIGH or CRITICAL blood pressure   -> problem: hypertension
    HIGH glucose + HIGH HbA1c         -> problem: type 2 diabetes mellitus with hyperglycemia
      (ICD-10-CM default: when report does not state Type 1 explicitly, ALWAYS use "type 2")
    LOW hemoglobin + LOW hematocrit   -> problem: anemia
- Only generate a measurement-descriptor problem when no clinical diagnosis can be
  concluded from the value and its clinical context.

CKD STAGING (when eGFR is reported):
- eGFR >= 90 -> stage 1; eGFR 60-89 -> stage 2; eGFR 45-59 -> stage 3a;
  eGFR 30-44 -> stage 3b; eGFR 15-29 -> stage 4; eGFR < 15 -> stage 5
- ALWAYS include the sub-stage (3a/3b) in the problem name when eGFR is reported.

COMBINATION CONDITIONS AND MULTI-ORGAN INVOLVEMENT:
- When ABNORMAL RESULTS contains flagged values from two or more different organ
  systems or sections, identify whether a single primary condition drives that
  multi-organ pattern. If yes:
  (a) Generate the primary condition as a standalone problem
  (b) Generate one problem per primary-condition plus each affected-organ combination
  (c) Generate each component condition as its own standalone problem

INDEPENDENT CONDITIONS -- no implicit causation:
- When the report shows abnormal results suggesting two DIFFERENT named conditions,
  treat them as SEPARATE independent problems UNLESS the note EXPLICITLY states
  a causal link using words like "caused by", "secondary to", "due to".

MANIFESTATION QUERIES:
- When a condition is a downstream manifestation of another condition, generate the
  manifestation as its own problem AND include a query with "in diseases classified
  elsewhere" -- this is how ICD-10-CM titles etiology-manifestation codes.

QUERY CONSTRUCTION -- CRITICAL:
- Queries are sent as literal text to a vector database of ICD-10-CM code TITLES.
- PRIMARY RULE: Write the DISEASE or DISORDER name -- never the measurement description.
- Build a specificity ladder per problem using 2-4 queries:
  * Query 1: most specific disease name plus key modifier
  * Query 2: disease name combined with affected organ or system
  * Query 3: always include one query with "other" prepended/appended
  * Query 4 (optional): broader synonym or unspecified fallback
- All queries: lowercase; no punctuation; no ICD codes; 2-20 words each

CONFIDENCE:
- high: named diagnosis stated clearly or directly concluded from the report
- medium: clearly supported by one or more flagged abnormal results
- low: indirect or weak inference only

Return JSON now.
"""

_CONF_SET = {"high", "medium", "low"}


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def _normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def plan_queries(
    *,
    note: str,
    openrouter_api_key: str,
    model: str,
) -> List[Dict[str, Any]]:
    user_msg = f"""Clinical note:\n{note}\n\nReturn STRICT JSON now."""
    logger.info("Planner: sending %d-char clinical note to %s", len(note), model)

    text = chat_completion(
        model=model,
        api_key=openrouter_api_key,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=2500,
    )
    logger.info("Planner: raw response (%d chars): %.500s", len(text), text)

    cleaned = _strip_code_fences(text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(cleaned[s : e + 1])
            except json.JSONDecodeError:
                logger.warning("Planner: JSON parse failed after brace extraction.")
                data = {}
        else:
            logger.warning("Planner: no JSON braces found in response.")
            data = {}

    problems = data.get("problems", [])
    if not isinstance(problems, list):
        return []

    used_queries: set = set()
    out: List[Dict[str, Any]] = []

    for p in problems:
        if not isinstance(p, dict):
            continue

        prob = (p.get("problem") or "").strip()
        conf = (p.get("confidence") or "medium").strip().lower()
        if conf not in _CONF_SET:
            conf = "medium"

        queries = p.get("queries") or []
        if not prob or not isinstance(queries, list):
            continue

        norm_qs = []
        for q in queries:
            nq = _normalize_query(str(q))
            if nq and nq not in used_queries:
                norm_qs.append(nq)

        norm_qs = _dedupe_preserve_order(norm_qs)

        if prob and norm_qs:
            for q in norm_qs:
                used_queries.add(q)
            out.append({"problem": prob, "confidence": conf, "queries": norm_qs[:4]})

    return out