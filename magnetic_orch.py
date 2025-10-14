import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import Agent, AgentMagnetic

# ---------- SETUP ----------
kernel = Kernel()
kernel.add_service(OpenAIChatCompletion("gpt-4o-mini", api_key="YOUR_API_KEY"))

# Define multiple agents
sales_agent = Agent(
    name="SalesAgent",
    description="Handles product recommendations and sales queries."
)
support_agent = Agent(
    name="SupportAgent",
    description="Handles technical support, troubleshooting, and complaints."
)
marketing_agent = Agent(
    name="MarketingAgent",
    description="Handles promotions, campaigns, and marketing strategy."
)

# ---------- MAGNETIC ORCHESTRATION ----------
# Create a Magnetic agent orchestrator
magnetic_orchestrator = AgentMagnetic(
    kernel=kernel,
    agents=[sales_agent, support_agent, marketing_agent]
)

# Function to run orchestration
async def run_magnetic_orchestration(user_query: str):
    print(f"🔹 User Query: {user_query}\n")
    response = await magnetic_orchestrator.invoke(user_query)
    print("💬 Final Response from Magnetic Orchestration:")
    print(response)

# ---------- RUN ----------
if __name__ == "__main__":
    user_query = input("Enter your query for magnetic orchestration: ")
    asyncio.run(run_magnetic_orchestration(user_query))
