# ICD-10-CM MCP Server

A Model Context Protocol (MCP) server that exposes your ICD-10-CM RAG pipeline as three tools Claude Desktop can call directly.

## What it does

| Tool | When to use | Speed |
|---|---|---|
| `code_clinical_note` | Full clinical note or lab report → ICD-10 codes with rationale | ~15-30s |
| `search_icd10` | Quick semantic search for a condition or symptom | ~2-3s |

## Prerequisites

- Your Pinecone index already populated with ICD-10-CM embeddings
- OpenRouter API key
- Python 3.10+

## Installation

```bash
# Clone / copy this folder, then:
cd icd10_mcp
pip install -e .

# Copy and fill in your keys
cp .env .env.local  # or edit .env directly
```

Edit `.env`:
```env
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=icd10cm-2026
PINECONE_NAMESPACE=icd10cm_2026
OPENROUTER_API_KEY=your_key
PLANNER_MODEL=openai/gpt-4o-mini
SELECTOR_MODEL=openai/gpt-4o-mini
EMBED_MODEL=openai/text-embedding-3-small
```

## Test before connecting to Claude Desktop

```bash
python scripts/test_tools.py
```

All three tools should show ✅ before you proceed.

## Connect to Claude Desktop

Open your Claude Desktop config file:
- **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the `icd10-mcp` block (merge into existing `mcpServers` if you have others):

```json
{
  "mcpServers": {
    "icd10-mcp": {
      "command": "python",
      "args": ["-m", "icd10_mcp.server"],
      "cwd": "/absolute/path/to/icd10_mcp",
      "env": {
        "PINECONE_API_KEY": "your_pinecone_api_key",
        "PINECONE_INDEX_NAME": "icd10cm-2026",
        "PINECONE_NAMESPACE": "icd10cm_2026",
        "OPENROUTER_API_KEY": "your_openrouter_api_key",
        "PLANNER_MODEL": "openai/gpt-4o-mini",
        "SELECTOR_MODEL": "openai/gpt-4o-mini",
        "EMBED_MODEL": "openai/text-embedding-3-small"
      }
    }
  }
}
```

Restart Claude Desktop. You should see a 🔨 tools icon — click it to confirm the three tools appear.

## Deployment on Render

1. Push this repository to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repository
4. Set the build command: `pip install -e .`
5. Set the start command: `icd10-mcp --http`
6. Add environment variables in Render dashboard:
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
   - `PINECONE_NAMESPACE`
   - `OPENROUTER_API_KEY`
   - `PLANNER_MODEL`
   - `SELECTOR_MODEL`
   - `EMBED_MODEL`

Render will automatically use `render.yaml` for configuration.

## Project structure

```
icd10_mcp/
├── src/
│   └── icd10_mcp/
│       ├── __init__.py
│       ├── server.py            ← MCP server (entry point)
│       ├── planner.py           ← Stage 1: extract clinical problems
│       ├── retriever.py         ← Stage 2: Pinecone RRF search
│       ├── selector.py          ← Stage 3: LLM code selection
│       └── openrouter_client.py ← Shared API client
├── scripts/
│   └── test_tools.py            ← Smoke test all 3 tools
├── pyproject.toml
├── Dockerfile
├── .env.example
├── claude_desktop_config.json   ← Paste into Claude Desktop config
└── README.md
```

## Example Claude Desktop usage

Once connected, you can ask Claude:

> *"Here is a lab report, please give me the ICD-10 codes:"*  → Claude calls `code_clinical_note`

> *"What ICD-10 codes exist for COPD with acute exacerbation?"* → Claude calls `search_icd10`

## Architecture

```
Clinical Note
     │
     ▼
 [Planner]  ─── LLM (gpt-4o-mini) extracts problems + builds smart queries
     │
     ▼
 [Retriever] ── Pinecone vector search + RRF fusion + lexical re-ranking
     │
     ▼
 [Selector]  ── LLM (gpt-4o-mini) selects best codes with rationale
     │
     ▼
 ICD-10 Codes returned to Claude Desktop
```