import asyncio
import subprocess
import tempfile
import re
from typing import AsyncGenerator

from semantic_kernel.agents import (
    Agent,
    ChatCompletionAgent,
    MagenticOrchestration,
    StandardMagenticManager,
)
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = "https://etiasandboxaifoundry.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

agents_used = []

class CodeDebuggerAgent(Agent):
    def __init__(self):
        super().__init__(
            name="CodeDebuggerAgent",
            description="Executes Python code locally and returns the output or errors.",
        )

    async def get_response(self, task, **kwargs) -> ChatMessageContent:
        return await self._execute_code(task)

    async def invoke(self, task, **kwargs) -> ChatMessageContent:
        return await self._execute_code(task)

    async def invoke_stream(self, task, **kwargs) -> AsyncGenerator[ChatMessageContent, None]:
        yield await self._execute_code(task)

    async def _execute_code(self, task) -> ChatMessageContent:
        if isinstance(task, ChatMessageContent):
            task = task.content
        elif isinstance(task, list):
            task = " ".join(str(t) for t in task)
        elif not isinstance(task, str):
            task = str(task)

        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name,
                role="assistant",
                content="No Python code block found to execute."
            )

        code = code_blocks[0]

        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as temp_file:
                temp_file.write(code)
                temp_file.flush()
                result = subprocess.run(
                    ["python", temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            output = result.stdout or result.stderr
        except Exception as e:
            output = f"Error during execution: {str(e)}"

        return ChatMessageContent(
            name=self.name,
            role="assistant",
            content=f"Execution result:\n```\n{output}\n```"
        )

async def agents() -> list[Agent]:
    base_service = AzureChatCompletion(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    return [
        ChatCompletionAgent(
            name="ResearchAgent",
            description="A helpful assistant with access to web search. Ask it to perform web searches.",
            instructions="You are a Researcher. You find information without additional computation or quantitative analysis.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CoderAgent",
            description="A helpful assistant that writes and explains code to process and analyze data.",
            instructions="You solve questions using code. Please provide detailed analysis and computation process.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="DataAnalystAgent",
            description="An expert in interpreting data, generating insights, and performing statistical analysis.",
            instructions="You analyze datasets and extract meaningful insights using statistical methods.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="SustainabilityExpertAgent",
            description="An expert in estimating environmental impact, including CO2 emissions and energy usage.",
            instructions="You estimate carbon emissions and energy consumption based on compute resources and model usage.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="PresentationAgent",
            description="An assistant that formats results into tables, summaries, or slide-ready content.",
            instructions="You organize and present information clearly using tables, bullet points, and summaries.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CodeReviewerAgent",
            description="A code reviewer and debugger that identifies issues, suggests improvements, and ensures best practices.",
            instructions="You review code for bugs, performance issues, and adherence to best practices. Provide suggestions and fixes.",
            service=base_service,
        ),
        CodeDebuggerAgent(),
    ]

def agent_response_callback(message: ChatMessageContent) -> None:
    agents_used.append(message.name)
    print(f"\n**{message.name}**\n{message.content}\n")

async def main():
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
            "I am working for a large enterprise and I need to showcase Semantic Kernel is the way to go for orchestration. "
            "Write a Python script that calculates the most financially valuable opportunities for the enterprise. "
            "Then, run the code to verify it works correctly. "
            "Ensure the code is reviewed and debugged if needed. Execute the generated code to check for errors."
        ),
        runtime=runtime,
    )

    value = await orchestration_result.get()
    print(f"\n***** Final Result *****\n{value}")
    print("\nAgents involved:", agents_used)

    await runtime.stop_when_idle()

if __name__ == "__main__":
    asyncio.run(main())
