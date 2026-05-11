# AgentOps Python SDK

Tracing, prompt management, guardrails, and content management for AgentOps.

## Install

```bash
pip install https://github.com/Algorithma-se/agentops-sdk/releases/download/v0.2.0/agentops_sdk-0.2.0-py3-none-any.whl
```

Or in `requirements.txt`:

```
agentops_sdk @ https://github.com/Algorithma-se/agentops-sdk/releases/download/v0.2.0/agentops_sdk-0.2.0-py3-none-any.whl
```

## Configure

Set environment variables:

```bash
AGENTOPS_HOST=https://your-agentops-instance.example.com
AGENTOPS_PUBLIC_KEY=pk-lf-...
AGENTOPS_SECRET_KEY=sk-lf-...
```

Or pass them directly:

```python
from agentops import AgentOps

client = AgentOps(
    host="https://your-agentops-instance.example.com",
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
)
```

## Quick Start

```python
from agentops import AgentOps, observe

client = AgentOps()

# Tracing with the @observe decorator
@observe(as_type="generation")
def call_llm(messages):
    return openai_client.chat.completions.create(messages=messages)

@observe(as_type="span")
def search_docs(query):
    return vector_db.search(query)

# Create a trace with session grouping
trace = client.trace(
    name="my-agent",
    session_id="conversation-123",
    user_id="user-456",
    input={"query": "Hello"},
)

result = call_llm([{"role": "user", "content": "Hello"}])
trace.update(output=result)
client.flush()
```

## Managed Prompts

```python
prompt = client.get_prompt("my-system-prompt", label="production")
compiled = prompt.compile(name="Alice", role="assistant")
```

## Guardrails

```python
result = client.check_guardrails(
    user_input,
    stage="pre_input",
    agent_name="support-agent",
)
if result["action"] == "block":
    return "Sorry, I can't help with that."
```

## Content Management

```python
# Text content
content = client.get_content("faq-intro")

# Knowledge tables
data = client.get_table_data("product-catalog")
rows = client.search_table("product-catalog", "widget", limit=10)

# File assets
file = client.get_file("config-doc", download=True)
```

## Bootstrap Template

For a complete example with tracing and orchestrator logic, see:
`examples/agent_bootstrap.py`

## Running on Google Cloud Run (service-to-service auth)

If your AgentOps server runs on Cloud Run with the default invoker-locked
posture (i.e. **not** `allUsers`), public-API calls from another Cloud Run
service will be rejected by Cloud Run's IAM gate with a 401 — the SDK's
`Authorization: Basic pk:sk` header is opaque to Cloud Run, which requires
a Google-signed OIDC bearer.

Enable the SDK's built-in OIDC interceptor — no extra install, just an env var:

```bash
AGENTOPS_CLOUD_RUN_INVOKER_AUTH=true
# Optional — defaults to AGENTOPS_HOST
AGENTOPS_CLOUD_RUN_INVOKER_AUDIENCE=https://your-agentops.run.app
```

With the flag on, every outbound SDK request to your AgentOps host
carries an additional `X-Serverless-Authorization: Bearer <oidc>`
header. Cloud Run validates and strips it; the AgentOps container
only sees the normal Basic auth. Your runtime service account needs
`roles/run.invoker` on the AgentOps Cloud Run service — ask your
platform team to grant it.

The flag is **off by default**: SDK users outside GCP, on Cloud Run
services configured with `allUsers`, or behind a load balancer that
handles invoker auth themselves, don't need this and won't pay any
runtime cost. Calls to other hosts (OpenAI, Stripe, etc.) are never
touched — only requests matching the configured audience get the
header.
