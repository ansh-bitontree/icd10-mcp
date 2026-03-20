# ICD-10-CM MCP Server

A Model Context Protocol (MCP) server that provides ICD-10-CM medical coding tools for Claude Desktop, LangGraph, and web applications.

**🚀 New to this? Start with the [Quick Start Guide](QUICKSTART.md)**

## What it does

This server exposes two powerful tools for medical coding:

| Tool | Description | Use Case | Speed |
|---|---|---|---|
| `code_clinical_note` | Full 3-stage RAG pipeline: extracts clinical problems, retrieves matching codes, and selects the most accurate ICD-10 codes with rationale | Clinical notes, discharge summaries, lab reports | ~15-30s |
| `search_icd10` | Fast semantic vector search for ICD-10 codes | Quick lookups for specific conditions or symptoms | ~2-3s |

## 🚀 Quick Start - Use the Deployed Server

The easiest way to use this MCP server is to connect to the deployed instance. You only need your own OpenRouter API key.

**Deployed Server URL**: `https://icd10-mcp.onrender.com`

**Quick Config**: Copy `claude_desktop_config_example.json` and add your OpenRouter API key.

### For Claude Desktop Users

1. **Get your OpenRouter API key** from https://openrouter.ai/

2. **Open your Claude Desktop config file:**
   - **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

3. **Add this configuration** (merge into existing `mcpServers` if you have others):

```json
{
  "mcpServers": {
    "icd10-mcp": {
      "url": "https://icd10-mcp.onrender.com/sse",
      "transport": "sse",
      "headers": {
        "X-OpenRouter-API-Key": "sk-or-v1-your-key-here"
      }
    }
  }
}
```

4. **Restart Claude Desktop** and look for the 🔨 tools icon

**Note**: If the deployed server is not available or you prefer to run locally, see the "Self-Hosting" section below.

### For LangGraph / Python Applications

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Option 1: Connect to deployed server via SSE
import httpx

async def use_deployed_server():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://icd10-mcp.onrender.com/messages/",
            headers={
                "X-OpenRouter-API-Key": "sk-or-v1-your-key-here",
                "Content-Type": "application/json"
            },
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "code_clinical_note",
                    "arguments": {
                        "note": "Patient with type 2 diabetes and CKD stage 3",
                        "max_codes_per_problem": 2
                    }
                },
                "id": 1
            }
        )
        return response.json()

# Option 2: Use local MCP client
async def use_local_server():
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "icd10_mcp.server"],
        env={
            "OPENROUTER_API_KEY": "sk-or-v1-your-key-here",
            "PINECONE_API_KEY": "your-pinecone-key",
            "PINECONE_INDEX_NAME": "icd10cm-2026",
            "PINECONE_NAMESPACE": "icd10cm_2026"
        }
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Call the tool
            result = await session.call_tool(
                "code_clinical_note",
                arguments={
                    "note": "Patient with type 2 diabetes and CKD stage 3",
                    "max_codes_per_problem": 2
                }
            )
            return result
```

### For Web Applications (JavaScript/TypeScript)

```javascript
// Using fetch API to call MCP tools
async function codeClinicNote(clinicalNote) {
  const response = await fetch('https://icd10-mcp.onrender.com/messages/', {
    method: 'POST',
    headers: {
      'X-OpenRouter-API-Key': 'sk-or-v1-your-key-here',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'code_clinical_note',
        arguments: {
          note: clinicalNote,
          max_codes_per_problem: 2
        }
      },
      id: 1
    })
  });
  
  return await response.json();
}

// Example usage
const result = await codeClinicNote(
  "Patient presents with type 2 diabetes mellitus with diabetic nephropathy, stage 3 CKD"
);
console.log(result);

// Search for ICD-10 codes
async function searchICD10(query) {
  const response = await fetch('https://icd10-mcp.onrender.com/messages/', {
    method: 'POST',
    headers: {
      'X-OpenRouter-API-Key': 'sk-or-v1-your-key-here',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'search_icd10',
        arguments: {
          query: query,
          top_k: 10
        }
      },
      id: 2
    })
  });
  
  return await response.json();
}
```

### Using with n8n Workflow Automation

1. Add an **HTTP Request** node
2. Set the URL to: `https://icd10-mcp.onrender.com/messages/`
3. Method: `POST`
4. Add header: `X-OpenRouter-API-Key` with your key
5. Body (JSON):
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "code_clinical_note",
    "arguments": {
      "note": "{{$json.clinical_note}}",
      "max_codes_per_problem": 2
    }
  },
  "id": 1
}
```

## 💻 Self-Hosting (Advanced)

If you want to run your own instance locally or deploy your own server:

### Prerequisites

- Python 3.10+
- OpenRouter API key (get yours at https://openrouter.ai/)
- Pinecone account with ICD-10-CM embeddings index

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd icd10_mcp

# Install dependencies
pip install -e .

# Create .env file
touch .env
```

Edit `.env` and add your API keys:
```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=icd10cm-2026
PINECONE_NAMESPACE=icd10cm_2026

# OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model Configuration
PLANNER_MODEL=openai/gpt-4o-mini
SELECTOR_MODEL=openai/gpt-4o-mini
EMBED_MODEL=openai/text-embedding-3-small
```

### Test Your Local Installation

```bash
python scripts/test_tools.py
```

All tools should show ✅ before proceeding.

### Run Locally

**For Claude Desktop (stdio mode):**
```bash
python -m icd10_mcp.server
```

**For web/API access (HTTP/SSE mode):**
```bash
python -m icd10_mcp.server --http
```

### Connect Claude Desktop to Local Server

Edit your Claude Desktop config:

```json
{
  "mcpServers": {
    "icd10-mcp-local": {
      "command": "python",
      "args": ["-m", "icd10_mcp.server"],
      "cwd": "/absolute/path/to/icd10_mcp",
      "env": {
        "PINECONE_API_KEY": "your_pinecone_api_key_here",
        "PINECONE_INDEX_NAME": "icd10cm-2026",
        "PINECONE_NAMESPACE": "icd10cm_2026",
        "OPENROUTER_API_KEY": "your_openrouter_api_key_here",
        "PLANNER_MODEL": "openai/gpt-4o-mini",
        "SELECTOR_MODEL": "openai/gpt-4o-mini",
        "EMBED_MODEL": "openai/text-embedding-3-small"
      }
    }
  }
}
```

### Deploy Your Own Instance on Render

1. Fork this repository to your GitHub account
2. Create a new Web Service on [Render](https://render.com)
3. Connect your forked repository
4. Render will automatically detect `render.yaml` and configure the service
5. Add these environment variables in the Render dashboard:
   - `PINECONE_API_KEY` - Your Pinecone API key
   - `OPENROUTER_API_KEY` - Your OpenRouter API key
   - `PINECONE_INDEX_NAME` - (optional, defaults to icd10cm-2026)
   - `PINECONE_NAMESPACE` - (optional, defaults to icd10cm_2026)

Your server will be available at: `https://your-app-name.onrender.com`

**Note**: Users connecting to your deployed server will need to provide their own OpenRouter API key in the request headers.

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
├── claude_desktop_config.json   ← Paste into Claude Desktop config
└── README.md
```

## 📖 Usage Examples

### In Claude Desktop

Once connected, you can ask Claude:

> *"Here is a clinical note, please code it with ICD-10:"*
> 
> ```
> Patient presents with type 2 diabetes mellitus with diabetic nephropathy, 
> stage 3 chronic kidney disease, and uncontrolled hypertension.
> ```

Claude will call `code_clinical_note` and return structured ICD-10 codes with rationale.

> *"What ICD-10 codes exist for COPD with acute exacerbation?"*

Claude will call `search_icd10` for a quick lookup.

### Direct API Usage

**Code a clinical note:**
```bash
curl -X POST https://your-deployed-server.onrender.com/messages/ \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-API-Key: your_key_here" \
  -d '{
    "tool": "code_clinical_note",
    "arguments": {
      "note": "Patient with type 2 diabetes and CKD stage 3",
      "max_codes_per_problem": 2
    }
  }'
```

**Search for ICD-10 codes:**
```bash
curl -X POST https://your-deployed-server.onrender.com/messages/ \
  -H "Content-Type: application/json" \
  -H "X-OpenRouter-API-Key: your_key_here" \
  -d '{
    "tool": "search_icd10",
    "arguments": {
      "query": "essential hypertension",
      "top_k": 10
    }
  }'
```

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

## 🔑 API Key Management

### Where to Get API Keys

1. **OpenRouter API Key** (Required for all users)
   - Sign up at https://openrouter.ai/
   - Go to Keys section
   - Create a new API key
   - Add credits to your account (pay-as-you-go)

2. **Pinecone API Key** (Only required if self-hosting)
   - Sign up at https://www.pinecone.io/
   - Create a new project
   - Get your API key from the dashboard

### Cost Estimates (OpenRouter)

Using the default models:
- `code_clinical_note`: ~$0.01-0.03 per clinical note
- `search_icd10`: ~$0.001-0.002 per search

Costs depend on note length and complexity. See https://openrouter.ai/models for current pricing.

## ❓ FAQ

**Q: Do I need to deploy my own server?**
A: No! You can use the deployed server at `https://icd10-mcp.onrender.com` with just your OpenRouter API key.

**Q: Is my data secure?**
A: Clinical notes are processed in real-time and not stored. They're sent to OpenRouter's API for LLM processing. Review OpenRouter's privacy policy for details.

**Q: Can I use different LLM models?**
A: Yes! If self-hosting, you can change `PLANNER_MODEL` and `SELECTOR_MODEL` to any model available on OpenRouter.

**Q: What if the deployed server is slow or down?**
A: Render's free tier may have cold starts (10-30s delay). For production use, consider self-hosting or upgrading to a paid Render plan.

**Q: Can I use this for commercial purposes?**
A: Check the license file. For the OpenRouter API, review their terms of service.

## 🐛 Troubleshooting

### Claude Desktop doesn't show the tools

1. Check that your config file is valid JSON (use a JSON validator)
2. Ensure the server URL is correct and accessible
3. Verify your OpenRouter API key is valid
4. Restart Claude Desktop completely (quit and reopen)
5. Check Claude Desktop logs:
   - Mac: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

### "OpenRouter API key is required" error

- Make sure you've added the `X-OpenRouter-API-Key` header
- Verify your API key starts with `sk-or-v1-`
- Check that you have credits in your OpenRouter account

### "Pinecone connection failed" error

- This means the server's Pinecone configuration is incorrect
- If using the deployed server, contact the server administrator
- If self-hosting, verify your `PINECONE_API_KEY` and index name

### Slow response times

- First request may be slow due to cold start (Render free tier)
- Subsequent requests should be faster
- Consider self-hosting for consistent performance

### Tools return empty results

- Ensure your clinical note contains actual medical conditions
- Try the `search_icd10` tool first to verify the system is working
- Check that the Pinecone index contains ICD-10 data

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📧 Support

For issues or questions:
- Open a GitHub issue
- Check the troubleshooting section above
- Review MCP documentation: https://modelcontextprotocol.io/