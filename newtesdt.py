import asyncio
from semantic_kernel.agents import (
    Agent,
    ChatCompletionAgent,
    MagenticOrchestration,
    OpenAIAssistantAgent,
    StandardMagenticManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent


# Replace with your actual Azure OpenAI configuration
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = "https://etiasandboxaifoundry.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"  # or your specific API version


async def agents() -> list[Agent]:
    """Return a list of agents that will participate in the Magentic orchestration."""

    # Research agent using Azure OpenAI
    research_agent = ChatCompletionAgent(
        name="ResearchAgent",
        description="A helpful assistant with access to web search. Ask it to perform web searches.",
        instructions="You are a Researcher. You find information without additional computation or quantitative analysis.",
        service=AzureChatCompletion(
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
        ),
    )

    # Coder agent using Azure OpenAI
    coder_agent = ChatCompletionAgent(
        name="CoderAgent",
        description="A helpful assistant that writes and explains code to process and analyze data.",
        instructions="You solve questions using code. Please provide detailed analysis and computation process.",
        service=AzureChatCompletion(
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=AZURE_OPENAI_API_VERSION,
        ),
    )

    return [research_agent, coder_agent]


def agent_response_callback(message: ChatMessageContent) -> None:
    """Print messages from the agents."""
    print(f"\n**{message.name}**\n{message.content}\n")


async def main():
    """Main function to run the agents."""
    magentic_orchestration = MagenticOrchestration(
        members=await agents(),
        manager=StandardMagenticManager(
            chat_completion_service=AzureChatCompletion(
                api_key=AZURE_OPENAI_API_KEY,
                endpoint=AZURE_OPENAI_ENDPOINT,
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                api_version=AZURE_OPENAI_API_VERSION,
            )
        ),
        agent_response_callback=agent_response_callback,
    )

    runtime = InProcessRuntime()
    runtime.start()

    orchestration_result = await magentic_orchestration.invoke(
        task=(
            "I am preparing a report on the energy efficiency of different machine learning model architectures. "
            "Compare the estimated training and inference energy consumption of ResNet-50, BERT-base, and GPT-2 "
            "on standard datasets (e.g., ImageNet for ResNet, GLUE for BERT, WebText for GPT-2). "
            "Then, estimate the CO2 emissions associated with each, assuming training on an Azure Standard_NC6s_v3 VM "
            "for 24 hours. Provide tables for clarity, and recommend the most energy-efficient model "
            "per task type (image classification, text classification, and text generation)."
        ),
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(f"\n***** Final Result *****\n{value}")

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
