TypeError: Can't instantiate abstract class CodeDebuggerAgent without an implementation for abstract methods 'get_response', 'invoke_stream'

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

# =========================================================
# 🧩 Compatibility fixes for SK 1.37 (.message, .text, .thread)
# =========================================================
if not hasattr(ChatMessageContent, "message"):
    def _get_message(self): return getattr(self, "content", None)
    def _set_message(self, value): setattr(self, "content", value)
    ChatMessageContent.message = property(_get_message, _set_message)

if not hasattr(ChatMessageContent, "text"):
    def _get_text(self): return getattr(self, "content", None)
    def _set_text(self, value): setattr(self, "content", value)
    ChatMessageContent.text = property(_get_text, _set_text)

if not hasattr(ChatMessageContent, "thread"):
    @property
    def thread(self): return getattr(self, "_thread", None)
    @thread.setter
    def thread(self, value): setattr(self, "_thread", value)
    ChatMessageContent.thread = thread


# =========================================================
# 🔐 Azure OpenAI Configuration
# =========================================================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = "https://etiasandboxaifoundry.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

agents_used = []


# =========================================================
# 🧰 CodeDebuggerAgent (Executes + Reports back for fixes)
# =========================================================
class CodeDebuggerAgent(Agent):
    def __init__(self):
        super().__init__(
            name="CodeDebuggerAgent",
            description="Executes Python code locally, returns output, and reports errors for auto-fix.",
        )

    async def invoke(self, task, **kwargs) -> ChatMessageContent:
        return await self._execute_code(task, **kwargs)

    async def _execute_code(self, task, **kwargs) -> ChatMessageContent:
        thread = kwargs.get("thread", None)

        if isinstance(task, ChatMessageContent):
            task_text = task.content
            thread = getattr(task, "thread", thread)
        else:
            task_text = str(task)

        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task_text, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name, role="assistant",
                content="⚠️ No Python code block found to execute.", thread=thread
            )

        code = code_blocks[0].strip()

        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tf:
                tf.write(code)
                tf.flush()
                path = tf.name

            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=20
            )

            if result.returncode == 0:
                output = result.stdout.strip() or "✅ Code executed successfully (no output)."
                summary = "Execution successful."
            else:
                output = result.stderr.strip()
                summary = "Execution failed. Please fix the code."

        except subprocess.TimeoutExpired:
            output = "⏱️ Code execution timed out (20s limit)."
            summary = "Execution failed due to timeout."
        except Exception as e:
            output = f"❌ Runtime Error: {e}"
            summary = "Execution failed due to runtime exception."
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

        # Add structured context so another agent can fix code
        return ChatMessageContent(
            name=self.name,
            role="assistant",
            content=(
                f"{summary}\n\n💻 **Execution Output:**\n```\n{output}\n```"
                "\nIf there was an error, please analyze it and fix the Python code."
            ),
            thread=thread,
        )


# =========================================================
# 🤖 Define Other Agents
# =========================================================
async def agents() -> list[Agent]:
    base_service = AzureChatCompletion(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    return [
        ChatCompletionAgent(
            name="CoderAgent",
            description="Writes Python code to solve problems.",
            instructions="Write clean, correct Python code wrapped in ```python blocks```.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CodeReviewerAgent",
            description="Debugs and fixes Python errors reported by CodeDebuggerAgent.",
            instructions=(
                "If CodeDebuggerAgent reports an error, analyze the traceback, "
                "identify the cause, and return a corrected version of the code "
                "in a ```python``` block."
            ),
            service=base_service,
        ),
        CodeDebuggerAgent(),
    ]


# =========================================================
# 📡 Callback for Debug Output
# =========================================================
def agent_response_callback(msg: ChatMessageContent):
    name = getattr(msg, "name", "Unknown")
    text = getattr(msg, "content", "")
    agents_used.append(name)
    print(f"\n🔹 {name} says:\n{text}\n")


# =========================================================
# 🚀 Main Orchestration Run
# =========================================================
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

    try:
        orchestration_result = await orchestration.invoke(
            task=(
                "CoderAgent should generate Python code that calculates ROI for a list of investments. "
                "CodeDebuggerAgent should execute the code. "
                "If any errors occur, CodeReviewerAgent must correct and re-submit the fixed code for execution."
            ),
            runtime=runtime,
        )

        result = await orchestration_result.get()
        print(f"\n***** ✅ FINAL RESULT *****\n{getattr(result, 'content', str(result))}\n")
        print("Agents involved:", ", ".join(agents_used))

    except Exception as e:
        print(f"\n❌ Orchestration error: {e}\n")

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
