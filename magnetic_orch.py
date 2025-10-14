# full_magentic_azure.py
import asyncio
import logging
from html import escape

from semantic_kernel.kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.agents.orchestration.magentic_orchestration import (
    StandardMagenticManager,
    MagenticOrchestration,
)
from semantic_kernel.agents.agent import Agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------
# 1. Azure OpenAI Setup
# -----------------------
AZURE_OPENAI_ENDPOINT = "https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-api-key>"
AZURE_OPENAI_DEPLOYMENT = "<deployment-name>"
MODEL = "gpt-4"

kernel = Kernel()

azure_chat = AzureChatCompletion(
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    model=MODEL,
)

prompt_settings = PromptExecutionSettings()

# -----------------------
# 2. Define Agents
# -----------------------
class CodeWriterAgent(Agent):
    def __init__(self, kernel: Kernel):
        super().__init__("CodeWriter", description="Writes Python code based on task")
        self._kernel = kernel

    async def run(self, task: str) -> ChatMessageContent:
        prompt = f"Write a Python function for the following task:\n{task}"
        content = await azure_chat.get_chat_message_content(
            chat_history=None,
            prompt_execution_settings=prompt_settings,
            input_text=prompt,
        )
        return content


class CodeReviewerAgent(Agent):
    def __init__(self, kernel: Kernel):
        super().__init__("CodeReviewer", description="Reviews and suggests improvements for Python code")
        self._kernel = kernel

    async def run(self, code: str) -> ChatMessageContent:
        prompt = f"Review this Python code and suggest improvements:\n{code}"
        content = await azure_chat.get_chat_message_content(
            chat_history=None,
            prompt_execution_settings=prompt_settings,
            input_text=prompt,
        )
        return content

# -----------------------
# 3. Setup Magentic Manager
# -----------------------
manager = StandardMagenticManager(
    chat_completion_service=azure_chat,
    prompt_execution_settings=prompt_settings,
)

# -----------------------
# 4. Create Agents List
# -----------------------
agents = [
    CodeWriterAgent(kernel),
    CodeReviewerAgent(kernel),
]

# -----------------------
# 5. Magentic Orchestration
# -----------------------
magentic_orchestration = MagenticOrchestration(
    members=agents,
    manager=manager,
    name="CodeWriterReviewOrchestration",
    description="A Magentic orchestration where one agent writes code and the other reviews it.",
)

# -----------------------
# 6. Run Task
# -----------------------
async def main():
    task = ChatMessageContent(role="user", content="Write a Python function to calculate Fibonacci numbers recursively.")
    await magentic_orchestration.run(task)

if __name__ == "__main__":
    asyncio.run(main())
