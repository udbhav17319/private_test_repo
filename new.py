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
# Compatibility monkey-patch:
# ensure ChatMessageContent.message (and .text) exist and map to .content
# -----------------------------
if not hasattr(ChatMessageContent, "message"):
    # define a property that returns .content (string) and also allows setting it
    def _get_message(self):
        return getattr(self, "content", None)

    def _set_message(self, value):
        # keep both `content` and `message` consistent
        try:
            setattr(self, "content", value)
        except Exception:
            # fallback: set an attribute named 'message' if content can't be set
            object.__setattr__(self, "message", value)

    ChatMessageContent.message = property(_get_message, _set_message)

# also add .text for other code that expects it
if not hasattr(ChatMessageContent, "text"):
    def _get_text(self):
        return getattr(self, "content", None)
    def _set_text(self, value):
        try:
            setattr(self, "content", value)
        except Exception:
            object.__setattr__(self, "text", value)
    ChatMessageContent.text = property(_get_text, _set_text)


# ==============================
# ✅ AZURE OPENAI CONFIGURATION (unchanged)
# ==============================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = "https://etiasandboxaifoundry.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

agents_used = []


# ==============================
# CODE DEBUGGER AGENT
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
        # Normalize incoming task (support ChatMessageContent or raw string)
        if isinstance(task, ChatMessageContent):
            task_text = task.content
        elif isinstance(task, list):
            task_text = " ".join(str(t) for t in task)
        elif not isinstance(task, str):
            task_text = str(task)
        else:
            task_text = task

        # find python code block(s)
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", task_text, re.DOTALL)
        if not code_blocks:
            return ChatMessageContent(
                name=self.name,
                role="assistant",
                content="⚠️ No Python code block found to execute."
            )

        code = code_blocks[0].strip()

        try:
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tf:
                tf.write(code)
                tf.flush()
                temp_path = tf.name

            # execute using same python interpreter that's running this script
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=30,  # adjust timeout as needed
            )

            if result.returncode == 0:
                output = result.stdout.strip() or "✅ Code executed successfully (no output)."
            else:
                # show both stderr and stdout if helpful
                stderr = result.stderr.strip()
                stdout = result.stdout.strip()
                output_parts = []
                if stderr:
                    output_parts.append(f"STDERR:\n{stderr}")
                if stdout:
                    output_parts.append(f"STDOUT:\n{stdout}")
                output = "\n\n".join(output_parts) or f"⚠️ Process exited with code {result.returncode}."

        except subprocess.TimeoutExpired:
            output = "⏱️ Code execution timed out (30s)."
        except Exception as e:
            output = f"❌ Exception during execution: {e}"
        finally:
            # cleanup
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
# DEFINE AGENTS
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
            instructions="You are a researcher; gather relevant facts.",
            service=base_service,
        ),
        ChatCompletionAgent(
            name="CoderAgent",
            description="Writes and explains Python code.",
            instructions="Generate working Python code with comments.",
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
# CALLBACK (uses .content; monkey-patch covers any .message usage elsewhere)
# ==============================
def agent_response_callback(message: ChatMessageContent) -> None:
    agent_name = getattr(message, "name", "UnknownAgent")
    agent_content = getattr(message, "content", "")
    agents_used.append(agent_name)
    print(f"\n🔹 Agent: {agent_name}\n🗨️  Message:\n{agent_content}\n")


# ==============================
# MAIN
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
            "CoderAgent should write Python code that calculates top opportunities by ROI. "
            "Then CodeDebuggerAgent should execute the generated code and return the output."
        ),
        runtime=runtime,
    )

    value = await orchestration_result.get()
    # use .content; monkey-patch ensures old .message accessors work anywhere
    final_text = getattr(value, "content", str(value))
    print(f"\n***** ✅ FINAL RESULT *****\n{final_text}\n")
    print("Agents involved:", ", ".join(agents_used))

    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
