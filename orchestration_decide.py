from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import Agent, AgentGroupChat, AgentHandoff
import asyncio

# ---------- SETUP ----------
# Initialize the kernel
kernel = Kernel()

# Use either Azure OpenAI or OpenAI connector
# Example: OpenAI connector
kernel.add_service(OpenAIChatCompletion("gpt-4o-mini", api_key="YOUR_API_KEY"))

# Define sample agents
sales_agent = Agent(name="SalesAgent", description="Handles product recommendations and sales queries.")
support_agent = Agent(name="SupportAgent", description="Helps with troubleshooting and customer support issues.")

agents = [sales_agent, support_agent]

# ---------- FUNCTION TO DECIDE ORCHESTRATION ----------
async def decide_orchestration(user_query: str, kernel: Kernel) -> str:
    """
    Uses the LLM to decide which orchestration to use.
    Returns: 'group_chat' or 'handoff'
    """
    prompt = f"""
    You are an AI orchestrator. Based on the user query below,
    decide which orchestration strategy fits best:
    - 'group_chat': if multiple agents need to discuss or collaborate
    - 'handoff': if control should pass from one agent to another directly

    User query: "{user_query}"

    Respond with only one word: group_chat or handoff.
    """

    result = await kernel.services[0].complete(prompt)
    decision = result.strip().lower()

    if "handoff" in decision:
        return "handoff"
    return "group_chat"


# ---------- MAIN ORCHESTRATION RUNNER ----------
async def run_orchestration(user_query: str):
    decision = await decide_orchestration(user_query, kernel)
    print(f"🧠 LLM decided orchestration: {decision}\n")

    if decision == "group_chat":
        print("🤝 Running Group Chat orchestration...")
        group_chat = AgentGroupChat(kernel=kernel, agents=agents)
        response = await group_chat.invoke(user_query)
    else:
        print("🔄 Running Handoff orchestration...")
        handoff = AgentHandoff(kernel=kernel, agents=agents)
        response = await handoff.invoke(user_query)

    print("\n💬 Final Response:")
    print(response)


# ---------- RUN ----------
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    asyncio.run(run_orchestration(user_query))
