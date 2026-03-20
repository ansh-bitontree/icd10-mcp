# Quick Start Guide - ICD-10-CM MCP Server

This guide will help you connect to the ICD-10-CM MCP server in under 5 minutes.

## Step 1: Get Your OpenRouter API Key

1. Go to https://openrouter.ai/
2. Sign up or log in
3. Navigate to the "Keys" section
4. Click "Create Key"
5. Copy your API key (starts with `sk-or-v1-`)
6. Add credits to your account (minimum $5 recommended)

## Step 2: Configure Claude Desktop

### Option A: Use the Deployed Server (Easiest)

1. Open your Claude Desktop config file:
   - **Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add this configuration (or merge with existing `mcpServers`):

```json
{
  "mcpServers": {
    "icd10-mcp": {
      "url": "https://icd10-mcp.onrender.com/sse",
      "transport": "sse",
      "headers": {
        "X-OpenRouter-API-Key": "sk-or-v1-YOUR_KEY_HERE"
      }
    }
  }
}
```

3. Replace `sk-or-v1-YOUR_KEY_HERE` with your actual OpenRouter API key

### Option B: Run Locally

1. Clone this repository
2. Install dependencies: `pip install -e .`
3. Create `.env` file and add your keys (see README for format)
4. Use the config in `claude_desktop_config.json` (update paths)

## Step 3: Restart Claude Desktop

Completely quit and reopen Claude Desktop.

## Step 4: Verify Connection

1. Look for the 🔨 tools icon in Claude Desktop
2. Click it to see available tools
3. You should see:
   - `code_clinical_note`
   - `search_icd10`

## Step 5: Try It Out!

Ask Claude:

> "Search for ICD-10 codes for type 2 diabetes with kidney disease"

Or:

> "Here's a clinical note, please code it with ICD-10:
> 
> Patient presents with type 2 diabetes mellitus with diabetic nephropathy, stage 3 chronic kidney disease, and uncontrolled hypertension."

## Troubleshooting

### Tools don't appear
- Verify your JSON config is valid (use jsonlint.com)
- Check that you saved the config file
- Restart Claude Desktop again

### "API key required" error
- Make sure your OpenRouter API key is correct
- Verify you have credits in your OpenRouter account
- Check that the key starts with `sk-or-v1-`

### Slow first response
- The deployed server may have a cold start (10-30s)
- Subsequent requests will be faster
- This is normal for free-tier deployments

## Cost Information

Typical costs per request:
- Full clinical note coding: $0.01-0.03
- Quick ICD-10 search: $0.001-0.002

A $5 credit should handle 200-500 clinical notes.

## Next Steps

- Read the full README.md for advanced usage
- Try different clinical scenarios
- Integrate with your workflow (LangGraph, web apps, etc.)

## Support

Having issues? Check:
1. This guide's troubleshooting section
2. The main README.md FAQ section
3. Open a GitHub issue with details
