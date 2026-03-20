"""
ICD-10-CM MCP Server
====================
Supports TWO transports in one file:

  stdio    — Claude Desktop (local, no network needed)
  HTTP/SSE — Anyone: LangGraph, web apps, n8n, etc.

Two tools:
  1. code_clinical_note  — full 3-stage RAG pipeline
  2. search_icd10        — fast semantic vector search

Run modes:
  HTTP/SSE (Render deployment):
      python -m icd10_mcp.server --http

  stdio (Claude Desktop):
      python -m icd10_mcp.server
"""

import argparse
import asyncio
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from icd10_mcp.openrouter_client import OpenRouterEmbedder
from icd10_mcp.planner import plan_queries
from icd10_mcp.retriever import embed_text, get_index, pinecone_query, retrieve_all_candidates
from icd10_mcp.selector import select_codes

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load .env (local dev only — Render injects env vars directly) ─────────────
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX     = os.getenv("PINECONE_INDEX_NAME", "icd10cm-2026")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "icd10cm_2026")
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
PLANNER_MODEL      = os.getenv("PLANNER_MODEL", "openai/gpt-4o-mini")
SELECTOR_MODEL     = os.getenv("SELECTOR_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL        = os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")

# Callers must send this in X-API-Key header.
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
# Set it in Render dashboard. Leave blank only for local testing.
MCP_API_KEY = os.getenv("MCP_API_KEY", "")

PORT = int(os.getenv("PORT", "8000"))


# ── Lazy singletons ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedder() -> OpenRouterEmbedder:
    logger.info("Initialising embedder: %s", EMBED_MODEL)
    return OpenRouterEmbedder(api_key=OPENROUTER_API_KEY, model=EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_index():
    logger.info("Connecting to Pinecone index: %s", PINECONE_INDEX)
    return get_index(PINECONE_INDEX, PINECONE_API_KEY)


# ── MCP Server ────────────────────────────────────────────────────────────────

app = Server("icd10-mcp")


@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="code_clinical_note",
            description=(
                "Full ICD-10-CM coding pipeline for a clinical note or lab report. "
                "Extracts clinical problems, retrieves matching ICD-10 codes from a "
                "vector database, and selects the most accurate codes with rationale. "
                "Use this when you have a clinical note, discharge summary, or lab "
                "report and need accurate ICD-10-CM billing codes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": (
                            "The full clinical note, lab report, or discharge summary. "
                            "Include abnormal results, diagnoses, and relevant findings."
                        ),
                    },
                    "max_codes_per_problem": {
                        "type": "integer",
                        "description": "Max ICD codes per clinical problem (default: 2).",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["note"],
            },
        ),
        Tool(
            name="search_icd10",
            description=(
                "Fast ICD-10-CM semantic search. Returns top matching codes for a "
                "plain-text medical query. No LLM selector — faster but less precise "
                "than code_clinical_note. Good for quick lookups."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Medical condition or symptom in plain English. "
                            "e.g. 'essential hypertension', 'type 2 diabetes with CKD'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10, max: 25).",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def _handle_code_clinical_note(args: Dict[str, Any]) -> str:
    note: str = args.get("note", "").strip()
    if not note:
        return json.dumps({"error": "note is required and cannot be empty"})

    max_codes: int = int(args.get("max_codes_per_problem", 2))
    logger.info("code_clinical_note: note=%d chars", len(note))

    planned_problems = plan_queries(
        note=note,
        openrouter_api_key=OPENROUTER_API_KEY,
        model=PLANNER_MODEL,
    )
    if not planned_problems:
        return json.dumps({
            "error": "Planner could not extract any clinical problems.",
            "hint":  "Ensure the note contains diagnoses, findings, or lab results.",
        })
    logger.info("code_clinical_note: %d problems identified", len(planned_problems))

    merged, grouped, by_problem = retrieve_all_candidates(
        planned_problems=planned_problems,
        index=_get_index(),
        namespace=PINECONE_NAMESPACE,
        embed_model=_get_embedder(),
    )
    logger.info("code_clinical_note: %d unique candidates retrieved", len(merged))

    selection = select_codes(
        note=note,
        planned_problems=planned_problems,
        merged_candidates=merged,
        openrouter_api_key=OPENROUTER_API_KEY,
        model=SELECTOR_MODEL,
        max_codes_per_problem=max_codes,
        candidates_by_problem=by_problem,
    )

    return json.dumps(
        {"problems_identified": len(planned_problems), "results": selection.get("results", [])},
        ensure_ascii=False,
        indent=2,
    )


async def _handle_search_icd10(args: Dict[str, Any]) -> str:
    query: str = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"})

    top_k = min(int(args.get("top_k", 10)), 25)
    logger.info("search_icd10: %r top_k=%d", query, top_k)

    vector = embed_text(_get_embedder(), query)
    hits   = pinecone_query(
        index=_get_index(),
        namespace=PINECONE_NAMESPACE,
        vector=vector,
        top_k=top_k,
    )

    results = [
        {
            "code":        h["id"],
            "title":       (h.get("metadata") or {}).get("title", ""),
            "score":       round(h["score"], 4),
            "parent_code": (h.get("metadata") or {}).get("parent_code", ""),
            "level":       (h.get("metadata") or {}).get("level"),
        }
        for h in hits
    ]

    return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)


# ── MCP router ────────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        if name == "code_clinical_note":
            result = await _handle_code_clinical_note(arguments)
        elif name == "search_icd10":
            result = await _handle_search_icd10(arguments)
        else:
            result = json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as exc:
        logger.exception("Tool %s raised an exception", name)
        result = json.dumps({"error": str(exc)})

    return [TextContent(type="text", text=result)]


# ── HTTP / SSE layer ──────────────────────────────────────────────────────────

def _auth_ok(request: Request) -> bool:
    if not MCP_API_KEY:
        return True  # open access — local dev only
    return request.headers.get("X-API-Key", "") == MCP_API_KEY


sse = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    """
    SSE endpoint — MCP clients connect here.
    GET https://your-app.onrender.com/sse
    Header: X-API-Key: <your MCP_API_KEY>
    """
    if not _auth_ok(request):
        return JSONResponse(
            {"error": "Unauthorized — send your key in the X-API-Key header"},
            status_code=401,
        )
    logger.info("New SSE connection from %s", request.client)
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


async def handle_health(request: Request):
    """
    Health check — Render and UptimeRobot ping this.
    GET https://your-app.onrender.com/health
    """
    return JSONResponse({
        "status":  "ok",
        "server":  "icd10-mcp",
        "version": "0.1.0",
        "tools":   ["code_clinical_note", "search_icd10"],
        "auth":    "enabled" if MCP_API_KEY else "disabled (set MCP_API_KEY!)",
    })


# ── Starlette app ─────────────────────────────────────────────────────────────

starlette_app = Starlette(
    routes=[
        Route("/health",    endpoint=handle_health, methods=["GET"]),
        Route("/sse",       endpoint=handle_sse,    methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

starlette_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Entry points ──────────────────────────────────────────────────────────────

def _run_http() -> None:
    import uvicorn
    logger.info("Starting HTTP/SSE MCP server on 0.0.0.0:%d", PORT)
    uvicorn.run(starlette_app, host="0.0.0.0", port=PORT)


def _run_stdio() -> None:
    async def _go():
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())
    asyncio.run(_go())


def main() -> None:
    parser = argparse.ArgumentParser(description="ICD-10-CM MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run HTTP/SSE server (Render/LangGraph/web). Default: stdio (Claude Desktop).",
    )
    args = parser.parse_args()
    _run_http() if args.http else _run_stdio()


if __name__ == "__main__":
    main()