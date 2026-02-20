# RC-Baseline

Baseline probe runner and metrics for model behavior analysis.

## Consciousness Forge web page

This repository now includes a local web page route at `/consciousness-forge` served by `baseline.web_app`.

- The browser page never receives the Anthropic API key directly.
- Anthropic calls are proxied server-side through `POST /api/anthropic/messages`.
- Set `ANTHROPIC_API_KEY` in the server environment before use.

Run locally:

```bash
export ANTHROPIC_API_KEY="your-key"
python -m baseline.web_app
```

Then open: `http://127.0.0.1:8000/consciousness-forge`.
