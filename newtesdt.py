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
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"


async def agents() -> list[Agent]:
    """Return a list of agents that will participate in the Magentic orchestration."""

    base_service = AzureChatCompletion(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # 1️⃣ Research Agent
    research_agent = ChatCompletionAgent(
        name="ResearchAgent",
        description="Finds information and references from reliable academic or technical sources.",
        instructions="You are a research expert. Gather factual data, metrics, and papers relevant to the topic.",
        service=base_service,
    )

    # 2️⃣ Coder Agent
    coder_agent = ChatCompletionAgent(
        name="CoderAgent",
        description="Writes and explains code to process and analyze data.",
        instructions="You solve questions using Python code. Use Pandas, Matplotlib, and math where needed. Return code blocks with explanations.",
        service=base_service,
    )

    # 3️⃣ Data Analyst Agent
    data_analyst_agent = ChatCompletionAgent(
        name="DataAnalystAgent",
        description="Analyzes numerical or tabular data, performs comparisons, and interprets energy statistics.",
        instructions="You are a data scientist. When given data, summarize trends, compute metrics, and produce concise statistical conclusions.",
        service=base_service,
    )

    # 4️⃣ Environmental Analyst Agent
    env_agent = ChatCompletionAgent(
        name="EnvironmentalAnalystAgent",
        description="Estimates environmental impact, energy use, and CO2 emissions based on computation metrics.",
        instructions="You are an environmental expert. Estimate CO2 emissions and energy consumption using scientific assumptions. Use tables when helpful.",
        service=base_service,
    )

    # 5️⃣ Report Writer Agent
    report_agent = ChatCompletionAgent(
        name="ReportWriterAgent",
        description="Organizes and structures final results into a professional, readable report.",
        instructions="You are a report writer. Format results clearly with headings, bullet points, and tables. Maintain an academic tone.",
        service=base_service,
    )

    # 6️⃣ Reviewer Agent
    reviewer_agent = ChatCompletionAgent(
        name="ReviewerAgent",
        description="Reviews and validates the accuracy, coherence, and structure of the final report.",
        instructions="You are a reviewer. Check correctness, flow, and readability of outputs. Suggest any improvements or corrections.",
        service=base_service,
    )

    # 7️⃣ Visualizer Agent
    visualizer_agent = ChatCompletionAgent(
        name="VisualizerAgent",
        description="Creates visual representations of comparative results, such as tables or charts.",
        instructions="You are a visual data expert. Create ASCII tables or recommend chart types to summarize data effectively.",
        service=base_service,
    )

    return [
        research_agent,
        coder_agent,
        data_analyst_agent,
        env_agent,
        report_agent,
        reviewer_agent,
        visualizer_agent,
    ]


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
            "Prepare a report on the energy efficiency of different machine learning model architectures. "
            "Compare the estimated training and inference energy consumption of ResNet-50, BERT-base, and GPT-2 "
            "on standard datasets (e.g., ImageNet, GLUE, WebText). "
            "Estimate CO2 emissions assuming training on an Azure Standard_NC6s_v3 VM for 24 hours. "
            "Provide a detailed comparison table and identify the most energy-efficient model "
            "per task type (image classification, text classification, text generation)."
        ),
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(f"\n***** Final Result *****\n{value}")

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
