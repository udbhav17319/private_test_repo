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

# -----------------------------
# Compatibility: ensure old attributes exist if referenced
# -----------------------------
for attr in ["message", "text", "thread"]:
    if not hasattr(ChatMessageContent, attr):
        setattr(ChatMessageContent, attr, None)


# ==============================
# ✅ AZURE OPENAI CONFIGURATION (unchanged)
# ==============================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = "https://etiasandboxaifoundry.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

agents_used = []


# ==============================
# 💻 CODE DEBUGGER AGENT
# ==============================
class CodeDebuggerAgent(Agent):
    def __init__(self):
        super().__init__(
            name="CodeDebuggerAgent",
            description="Executes Python code locally and returns the output or errors.",
        )

    async def get_response(self, task, **kwargs) -> ChatMessageContent:
        return await self._execute_code(task, **kwargs)

    async def invoke(self, task, **kwargs) -> ChatMessageContent:
        return await self._execute_code(task, **kwargs)

    async def invoke_stream(self, task, **kwargs) -> AsyncGenerator[ChatMessageContent, None]:
        yield await self._execute_code(task, **kwargs)

    async def _execute_code(self, task, **kwargs) -> ChatMessageContent:
        """Extract Python code from message, execute it, and return output."""
        # Normalize input
        if isinstance(task, ChatMessageContent):
            task_text = task.content
            thread = getattr(task, "thread", None)
        else:
            task_text = str(task)
            thread = None

        # Extract code block
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task_text, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name,
                role="assistant",
                content="⚠️ No Python code block found to execute.",
                thread=thread,
            )

        code = code_blocks[0].strip()

        try:
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as temp_file:
                temp_file.write(code)
                temp_file.flush()
                temp_path = temp_file.name

            # Run the code safely
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
            content=f"💻 **Execution Result:**\n```\n{output}\n```",
            thread=thread,
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
            description="Finds relevant data and insights.",
            instructions="You are a researcher. Collect information needed for decision-making.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CoderAgent",
            description="Writes and explains Python code.",
            instructions="Generate working Python code with explanations.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CodeReviewerAgent",
            description="Reviews and fixes code.",
            instructions="Review for bugs, optimize, and ensure code clarity.",
            service=base_service,
        ),
        CodeDebuggerAgent(),
    ]


# ==============================
# 📡 CALLBACK HANDLER
# ==============================
def agent_response_callback(message: ChatMessageContent) -> None:
    agent_name = getattr(message, "name", "UnknownAgent")
    agent_content = getattr(message, "content", "")
    agents_used.append(agent_name)
    print(f"\n🔹 Agent: {agent_name}\n🗨️  Message:\n{agent_content}\n")


# ==============================
# 🚀 MAIN LOGIC
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

    orchestration_result = await orchestration.invoke(
        task=(
            "CoderAgent should write Python code to calculate top business opportunities by ROI. "
            "Then CodeDebuggerAgent should execute it locally and show results."
        ),
        runtime=runtime,
    )

    final = await orchestration_result.get()
    print(f"\n***** ✅ FINAL RESULT *****\n{getattr(final, 'content', str(final))}\n")
    print("Agents involved:", ", ".join(agents_used))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
