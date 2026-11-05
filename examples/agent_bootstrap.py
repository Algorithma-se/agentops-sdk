import os
import time
from agentops import AgentOps, observe
from agentops.talk import talk_turn_metadata

# ----------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ----------------------------------------------------------------------
# The SDK reads AGENTOPS_HOST, AGENTOPS_PUBLIC_KEY, and AGENTOPS_SECRET_KEY
# from environment variables. You can also pass them directly to AgentOps().
agentops = AgentOps()

# ----------------------------------------------------------------------
# 2. INSTRUMENTATION (Leaf steps)
# ----------------------------------------------------------------------

@observe(as_type="generation")
def call_llm(messages, model="gpt-4o"):
    """
    Records an LLM call as a 'Generation' in AgentOps.
    """
    # Replace with your actual LLM provider call (OpenAI, Anthropic, etc.)
    # Here we simulate a response.
    time.sleep(0.5)
    usage = {"input": 10, "output": 15}

    # Feed model + usage_details back onto this observation so AgentOps
    # can resolve a price and compute cost server-side — this is what
    # puts a "$0.0031" in Talk's provenance line under the reply. Without
    # this call the ingestion path has no `provided_model_name` and skips
    # cost calculation entirely (see AgentOps.update_current_observation).
    agentops.update_current_observation(model=model, usage_details=usage)

    return {
        "role": "assistant",
        "content": "Hello! I am your bootstrapped agent. How can I help?",
        "model": model,
        "usage": usage,
    }

@observe(as_type="span")
def search_knowledge_base(query: str):
    """
    Records a tool/retrieval step as a 'Span' in AgentOps.
    """
    time.sleep(0.2)
    return f"Result for '{query}': No specific docs found, using general knowledge."

# ----------------------------------------------------------------------
# 3. AGENT LOGIC (The Orchestrator)
# ----------------------------------------------------------------------

def run_agent(conversation_id: str, user_id: str, user_input: str, headers: dict = None):
    """
    A simple orchestrator that runs a trace for a single conversation turn.

    `headers` is whatever your web framework handed you for the inbound
    request (Flask's `request.headers`, FastAPI's `request.headers`, a
    plain dict, ...). When this agent is called by AgentOps Talk, it
    carries an `x-aos-talk-turn-id` header — pass it straight through so
    Talk can correlate this trace with the reply it sent, and show a
    trace fold + cost line for it. Talk works fine without this too; it
    just shows neither for that reply.
    """

    # 1. Fetch a managed prompt from the registry (optional but recommended)
    # prompt = agentops.get_prompt("my-agent-system-prompt", label="production")
    # system_message = prompt.compile()
    system_message = "You are a helpful assistant."

    # 2. Start a Trace
    # Use session_id to group turns in the 'Conversations' UI.
    trace = agentops.trace(
        name="my-bootstrap-agent",
        session_id=conversation_id,
        user_id=user_id,
        input={"user_input": user_input},
        metadata=talk_turn_metadata(headers),
    )

    try:
        # Step: Search
        search_results = search_knowledge_base(user_input)

        # Step: LLM
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"Context: {search_results}"}
        ]
        response = call_llm(messages)

        # 3. Finalize Trace
        trace.update(output=response)
        return response

    except Exception as e:
        trace.update(output={"error": str(e)})
        raise
    finally:
        # 4. Flush to ensure events are sent immediately
        agentops.flush()

if __name__ == "__main__":
    # Example local run. The `headers` dict simulates what your web
    # framework would hand you from the actual HTTP request Talk sent —
    # in a real deployment you'd pass `request.headers` (or equivalent)
    # straight through instead of hardcoding this.
    print("🚀 Running bootstrap agent example...")
    result = run_agent(
        conversation_id="test-conv-001",
        user_id="user-123",
        user_input="What is AgentOps?",
        headers={"x-aos-talk-turn-id": "demo-talk-turn-id"},
    )
    print(f"Assistant: {result['content']}")
    print("✅ Done. Check your AgentOps dashboard for the trace.")
    print(
        "   Sent with a Talk correlation id — a reply that used this turn "
        "id will show a trace fold + cost line in Talk."
    )
