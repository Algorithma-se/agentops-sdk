import os
import time
from agentops import AgentOps, observe

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
    return {
        "role": "assistant",
        "content": "Hello! I am your bootstrapped agent. How can I help?",
        "model": model,
        "usage": {"input": 10, "output": 15}
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

def run_agent(conversation_id: str, user_id: str, user_input: str):
    """
    A simple orchestrator that runs a trace for a single conversation turn.
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
        input={"user_input": user_input}
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
    # Example local run
    print("🚀 Running bootstrap agent example...")
    result = run_agent(
        conversation_id="test-conv-001",
        user_id="user-123",
        user_input="What is AgentOps?"
    )
    print(f"Assistant: {result['content']}")
    print("✅ Done. Check your AgentOps dashboard for the trace.")
