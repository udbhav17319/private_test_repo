import asyncio
import logging
from html import escape
from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.magentic import (
    StandardMagenticManager,
    MagenticOrchestration,
)
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion  # Using OpenAI connector
from semantic_kernel.agents.runtime.core.cancellation_token import CancellationToken
from semantic_kernel.agents.runtime.core.core_runtime import CoreRuntime
from semantic_kernel.contents.chat_message_content import ChatMessageContent, AuthorRole
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

# Setup logging
logging.basicConfig(level=logging.INFO)


# -----------------------------
# 1. Initialize LLM
# -----------------------------
llm = OpenAIChatCompletion(
    model="gpt-4",  # GPT-4 for high-quality code generation
    api_key="YOUR_OPENAI_API_KEY",
)

prompt_settings = llm.instantiate_prompt_execution_settings()


# -----------------------------
# 2. Define Agents
# -----------------------------
class CodeWriterAgent(Agent):
    """Writes Python code for the given task."""
    def __init__(self):
        super().__init__(name="CodeWriter", description="Writes Python code for the given task.")


class CodeReviewerAgent(Agent):
    """Reviews Python code and suggests improvements."""
    def __init__(self):
        super().__init__(name="CodeReviewer", description="Reviews Python code and suggests improvements.")


writer_agent = CodeWriterAgent()
reviewer_agent = CodeReviewerAgent()


# -----------------------------
# 3. Define Manager
# -----------------------------
manager = StandardMagenticManager(chat_completion_service=llm, prompt_execution_settings=prompt_settings)


# -----------------------------
# 4. Setup Magentic Orchestration
# -----------------------------
orchestration = MagenticOrchestration(
    members=[writer_agent, reviewer_agent],
    manager=manager,
    name="CodeWriter-Reviewer Orchestration",
    description="A Magentic orchestration with a code writer and reviewer using LLM.",
)


# -----------------------------
# 5. Runtime simulation
# -----------------------------
async def main():
    runtime = CoreRuntime()

    async def exception_callback(exc: BaseException):
        logging.error(f"Exception: {exc}")

    async def result_callback(result: ChatMessageContent):
        print("\n--- FINAL CODE ---")
        print(result.content)

    # Prepare orchestration
    await orchestration._prepare(
        runtime,
        internal_topic_type="code_task_topic",
        exception_callback=exception_callback,
        result_callback=result_callback
    )

    # Start orchestration with a task
    task_description = "Write a Python function that takes a list of numbers and returns the sum of squares."
    task = ChatMessageContent(role=AuthorRole.USER, content=task_description)

    await orchestration._start(
        task=task,
        runtime=runtime,
        internal_topic_type="code_task_topic",
        cancellation_token=CancellationToken()
    )


# -----------------------------
# 6. Run the orchestration
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
