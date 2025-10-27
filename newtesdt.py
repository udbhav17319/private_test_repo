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
# 💻 CodeDebuggerAgent (Executes + passes errors upstream)
# =========================================================
class CodeDebuggerAgent(Agent):
    """Runs Python code locally and returns execution result or error report."""

    def __init__(self):
        super().__init__(
            name="CodeDebuggerAgent",
            description="Executes Python code locally and reports success or errors for automatic fixing.",
        )

    async def get_response(self, task, **kwargs):
        return await self._execute_code(task, **kwargs)

    async def invoke(self, task, **kwargs):
        return await self._execute_code(task, **kwargs)

    async def invoke_stream(self, task, **kwargs):
        yield await self._execute_code(task, **kwargs)

    async def _execute_code(self, task, **kwargs):
        """Extract and execute Python code safely, preserving thread context."""
        thread = kwargs.get("thread", None)

        if isinstance(task, ChatMessageContent):
            task_text = task.content
            thread = getattr(task, "thread", thread)
        else:
            task_text = str(task)

        # Extract code block
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task_text, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name, role="assistant",
                content="⚠️ No Python code block found to execute.",
                thread=thread,
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
                success = True
            else:
                output = result.stderr.strip() or result.stdout.strip()
                success = False

        except subprocess.TimeoutExpired:
            output = "⏱️ Code execution timed out (20s limit)."
            success = False
        except Exception as e:
            output = f"❌ Runtime Error: {e}"
            success = False
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

        if success:
            summary = "✅ Execution successful."
        else:
            summary = "❌ Execution failed. Please analyze and fix the Python code."

        return ChatMessageContent(
            name=self.name,
            role="assistant",
            content=f"{summary}\n\n💻 **Execution Output:**\n```\n{output}\n```",
            thread=thread,
        )


# =========================================================
# 🤖 Define All Agents
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
            description="Writes Python code to solve business problems.",
            instructions="Write complete and correct Python code inside ```python``` blocks.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CodeReviewerAgent",
            description="Fixes Python code based on errors from CodeDebuggerAgent.",
            instructions=(
                "If the previous code execution failed, analyze the error message, "
                "identify the issue, and return a corrected Python code block."
            ),
            service=base_service,
        ),
        CodeDebuggerAgent(),
    ]


# =========================================================
# 📡 Callback for Live Agent Logs
# =========================================================
def agent_response_callback(msg: ChatMessageContent):
    name = getattr(msg, "name", "Unknown")
    text = getattr(msg, "content", "")
    agents_used.append(name)
    print(f"\n🔹 {name} says:\n{text}\n")


# =========================================================
# 🚀 Orchestration Logic (self-healing loop)
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

    # Step 1: generate initial code
    current_task = "CoderAgent should generate Python code that calculates ROI for a list of investments."

    for iteration in range(5):  # limit retries
        print(f"\n===== 🌀 Iteration {iteration+1} =====")

        orchestration_result = await orchestration.invoke(
            task=current_task,
            runtime=runtime,
        )

        value = await orchestration_result.get()
        result_text = getattr(value, "content", str(value))

        # Check if execution was successful
        if "Execution successful" in result_text or "✅ Code executed successfully" in result_text:
            print("\n🎉 Code ran successfully! Exiting loop.\n")
            break
        else:
            # Ask reviewer to fix
            print("\n🔁 Code failed. Asking CodeReviewerAgent to fix it...\n")
            current_task = (
                "CodeReviewerAgent: Please fix the code based on this error report.\n"
                f"{result_text}"
            )

    print(f"\n***** ✅ FINAL RESULT *****\n{result_text}\n")
    print("Agents involved:", ", ".join(agents_used))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
