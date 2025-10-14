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
# os.environ["AZURE_OPENAI_ENDPOINT"] = "<your-endpoint>"
# os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "<your-deployment-name>"

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
# 🧠 3. Create Research Agent
# ============================================================
research_agent = ChatCompletionAgent(
    name="ResearchAgent",
    description="A helpful assistant with access to web search. Ask it to perform web searches.",
    instructions="You are a Researcher. You find information without additional computation or quantitative analysis.",
    service=azure_service,
)

# ============================================================
# 🧮 4. Create a Local Code Interpreter Placeholder
# ============================================================
# Instead of connecting to Azure Assistant API,
# we mock code interpreter configuration locally for now.

async def setup_local_coder_agent():
    # Local placeholder setup
    print("⚙️ Using local placeholder for code interpreter tool...")

    # Simulate code interpreter tool configuration
    code_interpreter_tool = {"name": "local_code_interpreter"}
    code_interpreter_tool_resources = {"execution": "local"}
    
    # Create dummy coder agent (using Azure ChatCompletion)
    coder_agent = ChatCompletionAgent(
        name="CoderAgent",
        description="A helpful assistant that writes and executes code to process and analyze data.",
        instructions=(
            "You solve questions using code. "
            "Simulate code execution locally. Provide detailed reasoning and pseudo-code results."
        ),
        service=azure_service,
    )

    return coder_agent, code_interpreter_tool, code_interpreter_tool_resources

# ============================================================
# 🗣️ 5. Agent Response Callback
# ============================================================
def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"\n**{message.name}**:\n{message.content}\n")

# ============================================================
# ⚙️ 6. Orchestration and Runtime
# ============================================================
async def main():
    coder_agent, code_interpreter_tool, code_interpreter_tool_resources = await setup_local_coder_agent()

    magentic_orchestration = MagenticOrchestration(
        members=[research_agent, coder_agent],
        manager=None,  # can define a manager later if needed
        agent_response_callback=agent_response_callback,
    )

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

    orchestration_result = await magentic_orchestration.invoke(
        task=task,
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(f"\n✅ Final result:\n{value}")

if __name__ == "__main__":
    asyncio.run(main())
