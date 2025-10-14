import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import Agent, AgentMagnetic

# -------------------- CONFIG --------------------
AZURE_OPENAI_ENDPOINT = "https://<YOUR_AZURE_OPENAI_RESOURCE>.openai.azure.com/"
AZURE_OPENAI_KEY = "<YOUR_AZURE_OPENAI_KEY>"
DEPLOYMENT_NAME = "<YOUR_DEPLOYMENT_NAME>"  # e.g., gpt-4o-mini
# -----------------------------------------------

# ---------- INIT KERNEL ----------
kernel = Kernel()

# Add Azure OpenAI chat service
azure_service = AzureChatCompletion(
    deployment_name=DEPLOYMENT_NAME,
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY
)
kernel.add_service(azure_service)

# ---------- DEFINE AGENTS ----------
code_writer_agent = Agent(
    name="CodeWriter",
    description="Writes clean, efficient, and well-documented Python code based on user requirements."
)

code_reviewer_agent = Agent(
    name="CodeReviewer",
    description="Reviews Python code for correctness, style, performance, and suggests improvements."
)

# ---------- MAGNETIC ORCHESTRATION ----------
magnetic_orchestrator = AgentMagnetic(
    kernel=kernel,
    agents=[code_writer_agent, code_reviewer_agent]
)

# ---------- RUN FUNCTION ----------
async def run_magnetic_code(user_query: str):
    print(f"🔹 User Query: {user_query}\n")
    response = await magnetic_orchestrator.invoke(user_query)
    print("💬 Final Response from Magnetic Orchestration:")
    print(response)

# ---------- MAIN ----------
if __name__ == "__main__":
    user_query = input("Enter your coding task or query: ")
    asyncio.run(run_magnetic_code(user_query))
