import os
import asyncio
from semantic_kernel.agents import ChatCompletionAgent, OpenAIAssistantAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.agents import MagenticOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime

# ============================================================
# 🧠 1. Azure OpenAI Configuration
# ============================================================
# Set these environment variables before running:
# os.environ["AZURE_OPENAI_API_KEY"] = "<your-azure-openai-key>"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "<your-endpoint>"  # e.g. "https://my-azure-openai.openai.azure.com/"
# os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "<your-deployment-name>"  # e.g. "gpt-4o"

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ============================================================
# 🧩 2. Define Azure Chat Completion service
# ============================================================
azure_service = AzureChatCompletion(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
)

# ============================================================
# 🧠 3. Create Research Agent (uses Azure OpenAI)
# ============================================================
research_agent = ChatCompletionAgent(
    name="ResearchAgent",
    description="A helpful assistant with access to web search. Ask it to perform web searches.",
    instructions="You are a Researcher. You find information without additional computation or quantitative analysis.",
    service=azure_service,
)

# ============================================================
# 🧮 4. Create a Coder Agent using Azure OpenAI Assistants API
# ============================================================
# Note: Azure OpenAI Assistant support mirrors OpenAI’s Assistants API.
# Make sure your Azure resource supports this model (e.g. gpt-4o or gpt-4-turbo).

client, model = await OpenAIAssistantAgent.setup_resources(
    api_key=AZURE_OPENAI_KEY,
    endpoint=AZURE_OPENAI_ENDPOINT,
)

code_interpreter_tool, code_interpreter_tool_resources = OpenAIAssistantAgent.configure_code_interpreter_tool()

definition = await client.beta.assistants.create(
    model=model,
    name="CoderAgent",
    description="A helpful assistant that writes and executes code to process and analyze data.",
    instructions="You solve questions using code. Please provide detailed analysis and computation process.",
    tools=code_interpreter_tool,
    tool_resources=code_interpreter_tool_resources,
)

coder_agent = OpenAIAssistantAgent(
    client=client,
    definition=definition,
)

# ============================================================
# 🗣️ 5. Agent Response Callback
# ============================================================
def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"**{message.name}**\n{message.content}")

# ============================================================
# ⚙️ 6. Orchestration Setup
# ============================================================
# ‘manager’ is optional; it can be a system prompt or a guiding agent.
# For simplicity, we let Magnetic Orchestration manage agents directly.

magentic_orchestration = MagenticOrchestration(
    members=[research_agent, coder_agent],
    manager=None,  # You can define a manager agent here if you like
    agent_response_callback=agent_response_callback,
)

# ============================================================
# 🚀 7. Run Orchestration
# ============================================================
runtime = InProcessRuntime()
runtime.start()

task = (
    "I am preparing a report on the energy efficiency of different machine learning model architectures. "
    "Compare the estimated training and inference energy consumption of ResNet-50, BERT-base, and GPT-2 "
    "on standard datasets (e.g., ImageNet for ResNet, GLUE for BERT, WebText for GPT-2). "
    "Then, estimate the CO2 emissions associated with each, assuming training on an Azure Standard_NC6s_v3 VM "
    "for 24 hours. Provide tables for clarity, and recommend the most energy-efficient model "
    "per task type (image classification, text classification, and text generation)."
)

async def main():
    orchestration_result = await magentic_orchestration.invoke(
        task=task,
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(f"\nFinal result:\n{value}")

if __name__ == "__main__":
    asyncio.run(main())
