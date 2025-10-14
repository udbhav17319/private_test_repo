import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import Agent
from typing import List, Callable
from collections import deque

# -------------------- CONFIG --------------------
AZURE_OPENAI_ENDPOINT = "https://<YOUR_AZURE_OPENAI_RESOURCE>.openai.azure.com/"
AZURE_OPENAI_KEY = "<YOUR_AZURE_OPENAI_KEY>"
DEPLOYMENT_NAME = "<YOUR_DEPLOYMENT_NAME>"  # e.g., gpt-4o-mini
# -----------------------------------------------

# ---------- INIT KERNEL ----------
kernel = Kernel()
kernel.add_service(AzureChatCompletion(
    deployment_name=DEPLOYMENT_NAME,
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY
))

# ---------- DEFINE AGENTS ----------
security_agent = Agent(
    name="SecurityAuditor",
    description="Reviews code for security flaws, hardcoded secrets, and OWASP Top 10 vulnerabilities."
)

reliability_agent = Agent(
    name="ReliabilityAgent",
    description="Audits code for reliability, fault tolerance, and error handling."
)

testing_agent = Agent(
    name="TestCoverageAgent",
    description="Ensures test coverage and suggests missing test cases."
)

product_owner = Agent(
    name="ProductOwner",
    description="Clarifies requirements and business objectives."
)

developer_agent = Agent(
    name="Developer",
    description="Breaks down requirements into user stories and technical tasks."
)

qa_agent = Agent(
    name="QATester",
    description="Writes test cases for user stories."
)

triage_agent = Agent(
    name="TriageAgent",
    description="Determines which agent should handle an issue (DB, Network, or escalate)."
)

db_agent = Agent(
    name="DBMitigationAgent",
    description="Handles database-related incidents."
)

net_agent = Agent(
    name="NetworkMitigationAgent",
    description="Handles network connectivity issues."
)

supplier_intel_agent = Agent(
    name="SupplierIntelAgent",
    description="Monitors supplier status and external risks."
)

alt_sourcing_agent = Agent(
    name="AltSourcingAgent",
    description="Finds alternative suppliers and evaluates feasibility."
)

ops_planner_agent = Agent(
    name="OpsPlannerAgent",
    description="Proposes revised production and logistics plans."
)

tri_agent = Agent(
    name="MagenticOneTriageAgent",
    description="Analyzes incidents and dynamically routes tasks to agents."
)

# ---------- CONCURRENT ORCHESTRATION ----------
async def concurrent_orchestration(code: str, agents: List[Agent]):
    tasks = [agent.invoke(code) for agent in agents]
    results = await asyncio.gather(*tasks)
    return results

# ---------- SEQUENTIAL ORCHESTRATION ----------
async def sequential_orchestration(code: str, agents: List[Agent]):
    results = []
    for agent in agents:
        result = await agent.invoke(code)
        results.append(result)
    return results

# ---------- GROUP CHAT ORCHESTRATION ----------
async def group_chat_orchestration(task: str, agents: List[Agent]):
    history = []
    current_input = task
    for _ in range(3):  # simple round-robin for demo
        for agent in agents:
            response = await agent.invoke(current_input)
            history.append((agent.name, response))
            current_input = response  # next agent sees previous output
    return history

# ---------- HANDOFF ORCHESTRATION ----------
async def handoff_orchestration(initial_task: str, triage_agent: Agent, db_agent: Agent, net_agent: Agent):
    # Simple triage logic for demo
    if "DB" in initial_task:
        selected_agent = db_agent
    elif "Network" in initial_task:
        selected_agent = net_agent
    else:
        selected_agent = triage_agent
    result = await selected_agent.invoke(initial_task)
    return result

# ---------- MAGNETIC-STYLE ORCHESTRATION ----------
async def magnetic_orchestration(input_text: str, triage_agent: Agent, agents: List[Agent]):
    # Triage first
    route_decision = await triage_agent.invoke(input_text)
    # Simple routing logic based on keywords
    selected_agents = []
    if "supplier" in route_decision.lower():
        selected_agents.append(supplier_intel_agent)
    if "fallback" in route_decision.lower() or "alternative" in route_decision.lower():
        selected_agents.append(alt_sourcing_agent)
    if "logistics" in route_decision.lower() or "production" in route_decision.lower():
        selected_agents.append(ops_planner_agent)
    # Default fallback
    if not selected_agents:
        selected_agents = agents
    # Run selected agents concurrently
    return await concurrent_orchestration(input_text, selected_agents)

# ---------- MAIN ----------
async def main():
    code_sample = """
    public async Task<string> GetUserProfileAsync(string userId)
    {
        var apiKey = "hardcoded-api-key";
        var client = new HttpClient();
        var response = await client.GetAsync("https://externalapi.com/user/" + userId);
        if (response.IsSuccessStatusCode) return await response.Content.ReadAsStringAsync();
        return null;
    }
    """
    print("===== Concurrent Orchestration =====")
    concurrent_results = await concurrent_orchestration(code_sample, [security_agent, reliability_agent, testing_agent])
    for r in concurrent_results:
        print(r)

    print("\n===== Sequential Orchestration =====")
    sequential_results = await sequential_orchestration(code_sample, [security_agent, reliability_agent, testing_agent])
    for r in sequential_results:
        print(r)

    print("\n===== Group Chat Orchestration =====")
    group_results = await group_chat_orchestration(
        "We need an online grocery system with same-day delivery and secure payments.",
        [product_owner, developer_agent, qa_agent]
    )
    for agent_name, r in group_results:
        print(f"{agent_name}: {r}")

    print("\n===== Handoff Orchestration =====")
    handoff_result = await handoff_orchestration(
        "Deployment failed due to DB connection timeout",
        triage_agent, db_agent, net_agent
    )
    print(handoff_result)

    print("\n===== Magnetic Orchestration =====")
    magnetic_result = await magnetic_orchestration(
        "Supplier in Taiwan halted shipments due to typhoon, revise production and logistics",
        tri_agent, [supplier_intel_agent, alt_sourcing_agent, ops_planner_agent]
    )
    for r in magnetic_result:
        print(r)

# Run
if __name__ == "__main__":
    asyncio.run(main())
