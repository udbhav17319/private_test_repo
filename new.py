import asyncio
import subprocess
import tempfile
import re
import os
import sys
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


# ==============================
# ✅ AZURE OPENAI CONFIGURATION
# ==============================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = "https://etiasandboxaifoundry.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

agents_used = []


# ==============================
# 🧠 LOCAL CODE EXECUTION AGENT
# ==============================
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
        """Extracts Python code blocks and executes them locally."""
        # Normalize task input
        if isinstance(task, ChatMessageContent):
            task = task.content
        elif isinstance(task, list):
            task = " ".join(str(t) for t in task)
        elif not isinstance(task, str):
            task = str(task)

        # Extract Python code block
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name,
                role="assistant",
                content="⚠️ No Python code block found to execute."
            )

        code = code_blocks[0].strip()

        try:
            # Write code to a temp file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as temp_file:
                temp_file.write(code)
                temp_file.flush()
                temp_path = temp_file.name

            # Execute locally
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=20
            )

            if result.returncode == 0:
                output = result.stdout.strip() or "✅ Code executed successfully (no output)."
            else:
                output = f"⚠️ Error (exit code {result.returncode}):\n{result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            output = "⏱️ Code execution timed out (20s limit)."
        except Exception as e:
            output = f"❌ Error during execution: {str(e)}"
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return ChatMessageContent(
            name=self.name,
            role="assistant",
            content=f"💻 **Execution Result:**\n```\n{output}\n```"
        )


# ==============================
# 🧩 DEFINE AGENTS
# ==============================
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
            description="Finds information for analysis.",
            instructions="You are a researcher; gather relevant data points and facts.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CoderAgent",
            description="Writes and explains Python code.",
            instructions="Generate correct, working Python code with explanations.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CodeReviewerAgent",
            description="Reviews and fixes code.",
            instructions="Detect bugs, inefficiencies, and improve code quality.",
            service=base_service,
        ),
        CodeDebuggerAgent(),
    ]


# ==============================
# 🧩 CLEAN CALLBACK FOR v1.37
# ==============================
def agent_response_callback(message: ChatMessageContent) -> None:
    """Safely print agent responses (v1.37 style)."""
    agent_name = getattr(message, "name", "UnknownAgent")
    agent_content = getattr(message, "content", str(message))
    agents_used.append(agent_name)

    print(f"\n🔹 Agent: {agent_name}")
    print(f"🗨️  Message:\n{agent_content}\n")


# ==============================
# 🚀 MAIN ORCHESTRATION
# ==============================
async def main():
    orchestration = MagenticOrchestration(
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

    result = await orchestration.invoke(
        task=(
            "CoderAgent should write Python code that calculates profitability ratios "
            "and returns the top 3 opportunities by ROI. "
            "Then CodeDebuggerAgent should execute it to verify results."
        ),
        runtime=runtime,
    )

    final_value = await result.get()
    print(f"\n***** ✅ FINAL RESULT *****\n{final_value.content}\n")
    print("Agents involved:", ", ".join(agents_used))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
